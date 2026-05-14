#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from urllib.parse import urlparse

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from warp_as_history.camera_warp import (
    CAMERA_CONTROL_DEFAULT_MESH_BREAK_MODE,
    CAMERA_CONTROL_DEFAULT_MESH_DEPTH_RTOL,
    CAMERA_CONTROL_DEFAULT_MESH_NORMAL_TOL_DEG,
    CAMERA_CONTROL_DEFAULT_WARP_INVISIBLE_FILL,
    CAMERA_CONTROL_DEFAULT_WARP_RENDER_MODE,
    CAMERA_CONTROL_DEFAULT_WARP_TARGET_FILL_MIN_NEIGHBORS,
    CAMERA_CONTROL_DEFAULT_WARP_TARGET_FILL_RADIUS,
    CAMERA_CONTROL_PI3_PIXEL_LIMIT,
    CAMERA_CONTROL_PROMPT_TRIGGER,
    Pi3XWarpRenderer,
    Pi3XWarpRendererConfig,
    center_crop_resize_first_frame,
    se3_inverse,
)
from warp_as_history.training import core as opt
from warp_as_history.training.utils import detach_tree


ONLINE_VIDEO_COLUMNS = ("video", "video_url", "url", "video_path", "path")
ONLINE_PROMPT_COLUMNS = ("prompt", "prompts", "caption", "text")
ONLINE_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _online_infer_column(columns, requested, candidates, label):
    if requested:
        if requested not in columns:
            raise KeyError(f"Requested online {label} column {requested!r} is missing from CSV header {list(columns)}.")
        return requested
    for name in candidates:
        if name in columns:
            return name
    raise KeyError(f"Could not infer online {label} column from CSV header {list(columns)}.")


def add_online_prompt_trigger(prompt, trigger=None):
    prompt = str(prompt or "").strip()
    trigger = str(CAMERA_CONTROL_PROMPT_TRIGGER if trigger is None else trigger).strip()
    if not trigger:
        return prompt
    if prompt.startswith(trigger):
        return prompt
    return f"{trigger} {prompt}".strip()


def normalize_online_training_dataframe(df, exact_args):
    columns = list(df.columns)
    video_column = _online_infer_column(
        columns,
        str(getattr(exact_args, "online_video_column", "") or ""),
        ONLINE_VIDEO_COLUMNS,
        "video",
    )
    prompt_column = _online_infer_column(
        columns,
        str(getattr(exact_args, "online_prompt_column", "") or ""),
        ONLINE_PROMPT_COLUMNS,
        "prompt",
    )
    prompt_trigger = str(getattr(exact_args, "online_prompt_trigger", CAMERA_CONTROL_PROMPT_TRIGGER) or "")
    rows = []
    for row_index, (_, row) in enumerate(df.iterrows()):
        base = row.to_dict()
        raw_prompt = str(base.get(prompt_column, ""))
        base["id"] = str(base.get("id") or f"online_{row_index:06d}")
        base["online_row_index"] = int(row_index)
        base["video_path"] = str(base[video_column])
        base["prompt_raw"] = raw_prompt
        base["prompt"] = add_online_prompt_trigger(raw_prompt, prompt_trigger)
        rows.append(base)
    normalized = df.__class__(rows)
    meta = {
        "video_column": video_column,
        "prompt_column": prompt_column,
        "prompt_trigger": prompt_trigger,
        "rows": len(rows),
    }
    return normalized, meta


def _online_is_uri(value):
    parsed = urlparse(str(value))
    return bool(parsed.scheme) and parsed.scheme not in {"", "file"}


def resolve_online_video_ref(value, data_root):
    text = str(value).strip()
    if _online_is_uri(text):
        return text
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = Path(data_root) / path
    return path


def _iter_online_image_files(path):
    return sorted(p for p in Path(path).iterdir() if p.suffix.lower() in ONLINE_IMAGE_EXTS)


def load_online_video_frames(ref, *, height, width, frame_stride=1, max_video_frames=0):
    frame_stride = max(1, int(frame_stride))
    max_video_frames = int(max_video_frames)
    frames = []
    if isinstance(ref, Path) and ref.is_dir():
        for src_idx, path in enumerate(_iter_online_image_files(ref)):
            if src_idx % frame_stride != 0:
                continue
            frame = Image.open(path).convert("RGB")
            frames.append(center_crop_resize_first_frame(frame, int(height), int(width)))
            if max_video_frames > 0 and len(frames) >= max_video_frames:
                break
    else:
        reader = imageio.get_reader(str(ref))
        try:
            for src_idx, array in enumerate(reader):
                if src_idx % frame_stride != 0:
                    continue
                frame = Image.fromarray(np.asarray(array)).convert("RGB")
                frames.append(center_crop_resize_first_frame(frame, int(height), int(width)))
                if max_video_frames > 0 and len(frames) >= max_video_frames:
                    break
        finally:
            reader.close()
    if not frames:
        raise ValueError(f"No frames decoded from online training video {ref}.")
    return frames


def online_pil_to_tensor(frame):
    arr = np.asarray(frame.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor * 2.0 - 1.0


def online_tensor_video_to_pil_frames(video):
    if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 3:
        raise ValueError(f"Expected online warp video tensor [1, 3, T, H, W], got {tuple(video.shape)}.")
    arr = video[0].detach().float().cpu().clamp(-1.0, 1.0)
    arr = ((arr + 1.0) * 127.5).round().to(torch.uint8)
    arr = arr.permute(1, 2, 3, 0).numpy()
    return [Image.fromarray(frame, mode="RGB") for frame in arr]


def online_mask_tensor_to_pil_frames(mask):
    if mask.ndim != 5 or mask.shape[0] != 1 or mask.shape[1] != 1:
        raise ValueError(f"Expected online visibility mask tensor [1, 1, T, H, W], got {tuple(mask.shape)}.")
    arr = mask[0, 0].detach().float().cpu().clamp(0.0, 1.0)
    arr = (arr * 255.0).round().to(torch.uint8).numpy()
    return [Image.fromarray(frame, mode="L") for frame in arr]


def subset_online_geometry(full_geometry, keyframe_indices):
    if not keyframe_indices:
        raise ValueError("Online warp rendering requires at least one keyframe.")
    keyframe_geometries = full_geometry["keyframe_geometries"]
    selected = [keyframe_geometries[int(idx)] for idx in keyframe_indices]
    latest = selected[-1]
    geometry = dict(full_geometry)
    geometry["intrinsic"] = latest["intrinsic"]
    geometry["keyframe_count"] = len(selected)
    geometry["keyframe_geometries"] = selected
    geometry["preserve_pi3x_keyframe_points"] = True
    geometry["render_height"] = latest["render_height"]
    geometry["render_width"] = latest["render_width"]
    geometry["source_pose"] = latest["source_pose"]
    geometry["source_rgb_u8"] = latest["source_rgb_u8"]
    return geometry


def online_relative_poses(full_geometry, source_pose, target_indices):
    keyframe_geometries = full_geometry["keyframe_geometries"]
    target_world = np.stack(
        [np.asarray(keyframe_geometries[int(idx)]["source_pose"], dtype=np.float32) for idx in target_indices],
        axis=0,
    )
    source_inv = se3_inverse(np.asarray(source_pose, dtype=np.float32)[None])[0]
    return np.einsum("ij,tjk->tik", source_inv.astype(np.float32, copy=False), target_world).astype(np.float32)


def online_renderer_config_from_args(args):
    return Pi3XWarpRendererConfig(
        pi3_pixel_limit=int(getattr(args, "online_pi3_pixel_limit", CAMERA_CONTROL_PI3_PIXEL_LIMIT)),
        conf_threshold=float(getattr(args, "online_pi3_conf_threshold", 0.1)),
        depth_edge_rtol=float(getattr(args, "online_pi3_depth_edge_rtol", 0.03)),
        mesh_samples_per_axis=int(getattr(args, "online_mesh_samples_per_axis", 4)),
        render_mode=str(getattr(args, "online_render_mode", CAMERA_CONTROL_DEFAULT_WARP_RENDER_MODE)),
        target_fill_radius=int(getattr(args, "online_target_fill_radius", CAMERA_CONTROL_DEFAULT_WARP_TARGET_FILL_RADIUS)),
        target_fill_min_neighbors=int(
            getattr(args, "online_target_fill_min_neighbors", CAMERA_CONTROL_DEFAULT_WARP_TARGET_FILL_MIN_NEIGHBORS)
        ),
        mesh_break_mode=str(getattr(args, "online_mesh_break_mode", CAMERA_CONTROL_DEFAULT_MESH_BREAK_MODE)),
        mesh_depth_rtol=float(getattr(args, "online_mesh_depth_rtol", CAMERA_CONTROL_DEFAULT_MESH_DEPTH_RTOL)),
        mesh_normal_tol_deg=float(
            getattr(args, "online_mesh_normal_tol_deg", CAMERA_CONTROL_DEFAULT_MESH_NORMAL_TOL_DEG)
        ),
    )


class OnlineWarpTrainingCache:
    def __init__(self, rows, exact_args, device):
        self.rows = [row.to_dict() if hasattr(row, "to_dict") else dict(row) for row in rows]
        self.exact_args = exact_args
        self.device = torch.device(device)
        self.renderer = Pi3XWarpRenderer(online_renderer_config_from_args(exact_args))
        self.records = {}
        self._precompute_all()
        self.renderer._pi3x_runtime = None
        opt.clean_memory()

    def _precompute_all(self):
        for row_index, row in enumerate(self.rows):
            ref = resolve_online_video_ref(row["video_path"], getattr(self.exact_args, "data_root", "."))
            forward_frames = load_online_video_frames(
                ref,
                height=int(self.exact_args.height),
                width=int(self.exact_args.width),
                frame_stride=int(getattr(self.exact_args, "online_frame_stride", 1)),
                max_video_frames=int(getattr(self.exact_args, "online_max_video_frames", 0)),
            )
            for direction, frames in (("forward", forward_frames), ("reverse", list(reversed(forward_frames)))):
                tensors = [online_pil_to_tensor(frame).unsqueeze(0) for frame in frames]
                print(
                    json.dumps(
                        {
                            "event": "online_warp_estimate_geometry",
                            "row_index": int(row_index),
                            "seq": str(row["id"]),
                            "direction": direction,
                            "frames": len(frames),
                            "video": str(ref),
                        }
                    ),
                    flush=True,
                )
                geometry = self.renderer.estimate_keyframe_geometry(tensors, device=self.device)
                self.records[(int(row_index), direction)] = {
                    "direction": direction,
                    "frames": frames,
                    "geometry": geometry,
                    "row": row,
                    "row_index": int(row_index),
                    "video_ref": str(ref),
                }

    def choose_direction(self, rng):
        if not bool(getattr(self.exact_args, "online_direction_augmentation", True)):
            return "forward"
        reverse_prob = float(getattr(self.exact_args, "online_direction_reverse_prob", 0.5))
        return "reverse" if rng.random() < reverse_prob else "forward"

    def sample_case(self, row_index, prepare_index):
        row_index = int(row_index)
        row = self.rows[row_index]
        rng = random.Random(
            opt.stable_seed_from_parts(int(self.exact_args.seed), "online_warp_training", row["id"], int(prepare_index))
        )
        direction = self.choose_direction(rng)
        prepared = self.records[(row_index, direction)]
        frames = prepared["frames"]
        n = len(frames)
        num_frames = int(self.exact_args.num_frames)
        if n < num_frames:
            raise ValueError(f"Online training video {prepared['video_ref']} has {n} frames, need {num_frames}.")

        first_prob = float(getattr(self.exact_args, "online_first_chunk_prob", 0.5))
        chunk_mode = "first" if rng.random() < first_prob or n <= num_frames else "later"
        if chunk_mode == "first":
            source_idx = rng.randint(0, n - num_frames)
            target_indices = list(range(source_idx, source_idx + num_frames))
            history_indices = []
            keyframe_indices = [source_idx]
            render_pose_indices = target_indices
            future_keyframe_indices = []
            drop_renderer_source = False
            keyframe_policy = "source_only"
            condition_frame = frames[source_idx]
        else:
            target_start = rng.randint(1, n - num_frames)
            target_indices = list(range(target_start, target_start + num_frames))
            max_history = min(int(getattr(self.exact_args, "online_max_history_frames", 19)), target_start)
            history_len = rng.randint(1, max(1, max_history))
            history_indices = list(range(target_start - history_len, target_start))
            future_keyframe_indices = []
            keyframe_policy = "history_only"
            if rng.random() < float(getattr(self.exact_args, "online_future_keyframe_prob", 0.5)):
                future_min = max(0, int(getattr(self.exact_args, "online_future_keyframes_min", 1)))
                future_max = max(future_min, int(getattr(self.exact_args, "online_future_keyframes_max", 2)))
                future_count = min(rng.randint(future_min, future_max), len(target_indices))
                if future_count > 0:
                    future_keyframe_indices = sorted(rng.sample(target_indices, future_count))
                keyframe_policy = "history_plus_future"
            keyframe_indices = sorted(set(history_indices + future_keyframe_indices))
            render_pose_indices = [keyframe_indices[-1], *target_indices]
            drop_renderer_source = True
            condition_frame = frames[history_indices[-1]]

        geometry = subset_online_geometry(prepared["geometry"], keyframe_indices)
        poses = online_relative_poses(prepared["geometry"], geometry["source_pose"], render_pose_indices)
        rendered = self.renderer.render_from_geometry(
            geometry,
            poses,
            height=int(self.exact_args.height),
            width=int(self.exact_args.width),
            device=self.device,
            invisible_fill_mode=str(
                getattr(self.exact_args, "online_invisible_fill", CAMERA_CONTROL_DEFAULT_WARP_INVISIBLE_FILL)
            ),
            render_mode=str(getattr(self.exact_args, "online_render_mode", CAMERA_CONTROL_DEFAULT_WARP_RENDER_MODE)),
            target_fill_radius=int(
                getattr(self.exact_args, "online_target_fill_radius", CAMERA_CONTROL_DEFAULT_WARP_TARGET_FILL_RADIUS)
            ),
            target_fill_min_neighbors=int(
                getattr(
                    self.exact_args,
                    "online_target_fill_min_neighbors",
                    CAMERA_CONTROL_DEFAULT_WARP_TARGET_FILL_MIN_NEIGHBORS,
                )
            ),
            mesh_break_mode=str(getattr(self.exact_args, "online_mesh_break_mode", CAMERA_CONTROL_DEFAULT_MESH_BREAK_MODE)),
        )
        warp_video = rendered["warp_video"]
        warp_mask = rendered["warp_visibility_mask"]
        if drop_renderer_source:
            warp_video = warp_video[:, :, 1:]
            warp_mask = warp_mask[:, :, 1:]
        warp_frames = online_tensor_video_to_pil_frames(warp_video)
        warp_mask_frames = online_mask_tensor_to_pil_frames(warp_mask)
        if len(warp_frames) != num_frames or len(warp_mask_frames) != num_frames:
            raise ValueError(
                f"Online warp rendered {len(warp_frames)} frames/{len(warp_mask_frames)} masks, need {num_frames}."
            )
        seq = f"{row['id']}:{direction}:{chunk_mode}:{int(prepare_index)}"
        return {
            "condition_frame": condition_frame,
            "direction": direction,
            "history_indices": history_indices,
            "keyframe_indices": keyframe_indices,
            "keyframe_policy": keyframe_policy,
            "future_keyframe_indices": future_keyframe_indices,
            "metadata": {
                "chunk_mode": chunk_mode,
                "direction": direction,
                "future_keyframe_indices": future_keyframe_indices,
                "history_indices": history_indices,
                "keyframe_indices": keyframe_indices,
                "keyframe_policy": keyframe_policy,
                "render_pose_indices": render_pose_indices,
                "row_index": int(row_index),
                "seq": seq,
                "target_indices": target_indices,
                "video": prepared["video_ref"],
                "warp_render_stats": rendered.get("warp_render_stats", {}),
            },
            "prompt": row["prompt"],
            "prompt_raw": row.get("prompt_raw", row["prompt"]),
            "row": row,
            "seq": seq,
            "target_frames": [frames[idx] for idx in target_indices],
            "target_indices": target_indices,
            "warp_frames": warp_frames,
            "warp_mask_frames": warp_mask_frames,
        }


def build_online_warp_training_cache(df, exact_args, device):
    rows = [row for _, row in df.iterrows()]
    return OnlineWarpTrainingCache(rows, exact_args, device)

def prompt_cache_key(exact_args, prompt):
    payload = {
        "base_model_path": str(exact_args.base_model_path),
        "prompt": str(prompt),
        "num_videos_per_prompt": 1,
        "max_sequence_length": 512,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def encode_prompt_cached(pipe, prompt, exact_args, device, cache_dir, memory_cache):
    key = prompt_cache_key(exact_args, prompt)
    if key in memory_cache:
        cached = memory_cache[key]
        return cached["prompt_embeds"], "memory"

    cache_path = None
    if cache_dir:
        cache_path = Path(cache_dir) / f"{key}.pt"
        if cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu")
            prompt_embeds = payload["prompt_embeds"].to(device=device, dtype=pipe.transformer.dtype)
            memory_cache[key] = {"prompt_embeds": prompt_embeds}
            return prompt_embeds, "disk"

    with torch.no_grad():
        prompt_embeds, _negative_prompt_embeds = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=opt.NEGATIVE_PROMPT,
            do_classifier_free_guidance=False,
            num_videos_per_prompt=1,
            max_sequence_length=512,
            device=device,
        )
    prompt_embeds = prompt_embeds.to(pipe.transformer.dtype)
    memory_cache[key] = {"prompt_embeds": prompt_embeds.detach()}

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "prompt_embeds": prompt_embeds.detach().cpu(),
                "meta": {
                    "prompt": str(prompt),
                    "base_model_path": str(exact_args.base_model_path),
                    "max_sequence_length": 512,
                },
            },
            cache_path,
        )
    return prompt_embeds, "encode"


def _restore_optional_attr(obj, name, had_value, old_value):
    if had_value:
        setattr(obj, name, old_value)
    elif hasattr(obj, name):
        delattr(obj, name)


def prepare_online_warp_item(pipe, row_index, exact_args, device, mean, std, keep_frames, cache_dir, memory_prompt_cache, prepare_index=0):
    online_cache = getattr(exact_args, "online_warp_cache", None)
    if online_cache is None:
        raise ValueError("Training cache is missing.")
    case = online_cache.sample_case(row_index, prepare_index)
    seq = case["seq"]
    prompt_text = str(case["prompt"])
    first_frame = case["condition_frame"]
    target_frames = case["target_frames"]
    history_frames = case["warp_frames"]
    mask_frames = case["warp_mask_frames"]

    had_extra_mask = hasattr(exact_args, "history_visibility_extra_mask_frames")
    old_extra_mask = getattr(exact_args, "history_visibility_extra_mask_frames", None)
    exact_args.history_visibility_extra_mask_frames = mask_frames

    try:
        with torch.no_grad():
            target_latents = opt.encode_video_latents(pipe, target_frames, exact_args, device, mean, std).detach()
            cached_prompt_embeds, prompt_cache_status = encode_prompt_cached(
                pipe,
                prompt_text,
                exact_args,
                device,
                cache_dir,
                memory_prompt_cache,
            )
            prompt_embeds, image_latents, fake_image_latents, video_latents = opt.prepare_condition(
                pipe,
                first_frame,
                prompt_text,
                exact_args,
                device,
                mean,
                std,
                history_frames=history_frames,
                prompt_embeds_override=cached_prompt_embeds,
            )
            histories = opt.make_histories(
                pipe,
                image_latents,
                fake_image_latents,
                exact_args,
                device,
                video_latents=video_latents,
                seq=seq,
            )
    finally:
        _restore_optional_attr(exact_args, "history_visibility_extra_mask_frames", had_extra_mask, old_extra_mask)

    item = {
        "seq": seq,
        "prompt": prompt_text,
        "prompt_raw": case.get("prompt_raw", prompt_text),
        "target_latents": target_latents,
        "prompt_embeds": prompt_embeds.detach(),
        "histories": detach_tree(histories),
        "prompt_cache_status": prompt_cache_status,
        "training": case["metadata"],
    }
    if keep_frames:
        item["target_frames"] = [frame.resize((exact_args.width, exact_args.height)) for frame in target_frames]
        item["history_frames"] = [frame.resize((exact_args.width, exact_args.height)) for frame in history_frames]
    print(json.dumps({"event": "online_warp_item_prepared", **case["metadata"]}), flush=True)
    return item


class LazyPreparedItems:
    def __init__(self, pipe, df, exact_args, device, mean, std, cache_dir):
        self.pipe = pipe
        self.rows = [row for _, row in df.iterrows()]
        self.exact_args = exact_args
        self.device = device
        self.mean = mean
        self.std = std
        self.cache_dir = cache_dir
        self.memory_prompt_cache = {}
        self.prompt_cache_status_counts = {}
        self.prepare_counter = 0
        if getattr(self.exact_args, "online_warp_cache", None) is None:
            self.exact_args.online_warp_cache = build_online_warp_training_cache(df, self.exact_args, self.device)

    def __len__(self):
        return len(self.rows)

    def _remember_status(self, status):
        self.prompt_cache_status_counts[status] = self.prompt_cache_status_counts.get(status, 0) + 1

    def get(self, idx):
        idx = int(idx)
        print(json.dumps({"event": "prepare_item_start", "index": idx, "seq": str(self.rows[idx]["id"])}), flush=True)
        self.prepare_counter += 1
        row_index = int(self.rows[idx]["online_row_index"]) if "online_row_index" in self.rows[idx] else idx
        item = prepare_online_warp_item(
            self.pipe,
            row_index,
            self.exact_args,
            self.device,
            self.mean,
            self.std,
            keep_frames=False,
            cache_dir=self.cache_dir,
            memory_prompt_cache=self.memory_prompt_cache,
            prepare_index=self.prepare_counter,
        )
        self._remember_status(item["prompt_cache_status"])
        print(
            json.dumps(
                {
                    "event": "prepare_item_done",
                    "index": idx,
                    "seq": item["seq"],
                    "prompt_cache_status": item["prompt_cache_status"],
                }
            ),
            flush=True,
        )
        return item
