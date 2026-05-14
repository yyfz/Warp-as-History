#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("XFORMERS_DISABLED", "1")

_CAMERA_WARP_SPEC = importlib.util.spec_from_file_location(
    "wah_camera_warp",
    REPO_ROOT / "warp_as_history" / "camera_warp.py",
)
if _CAMERA_WARP_SPEC is None or _CAMERA_WARP_SPEC.loader is None:
    raise ImportError("Could not load warp_as_history/camera_warp.py")
camera_warp = importlib.util.module_from_spec(_CAMERA_WARP_SPEC)
sys.modules[_CAMERA_WARP_SPEC.name] = camera_warp
_CAMERA_WARP_SPEC.loader.exec_module(camera_warp)

CAMERA_CONTROL_DEFAULT_MESH_BREAK_MODE = camera_warp.CAMERA_CONTROL_DEFAULT_MESH_BREAK_MODE
CAMERA_CONTROL_DEFAULT_MESH_DEPTH_RTOL = camera_warp.CAMERA_CONTROL_DEFAULT_MESH_DEPTH_RTOL
CAMERA_CONTROL_DEFAULT_MESH_NORMAL_TOL_DEG = camera_warp.CAMERA_CONTROL_DEFAULT_MESH_NORMAL_TOL_DEG
CAMERA_CONTROL_DEFAULT_WARP_INVISIBLE_FILL = camera_warp.CAMERA_CONTROL_DEFAULT_WARP_INVISIBLE_FILL
CAMERA_CONTROL_DEFAULT_WARP_RENDER_MODE = camera_warp.CAMERA_CONTROL_DEFAULT_WARP_RENDER_MODE
CAMERA_CONTROL_DEFAULT_WARP_TARGET_FILL_MIN_NEIGHBORS = camera_warp.CAMERA_CONTROL_DEFAULT_WARP_TARGET_FILL_MIN_NEIGHBORS
CAMERA_CONTROL_DEFAULT_WARP_TARGET_FILL_RADIUS = camera_warp.CAMERA_CONTROL_DEFAULT_WARP_TARGET_FILL_RADIUS
CAMERA_CONTROL_PI3_PIXEL_LIMIT = camera_warp.CAMERA_CONTROL_PI3_PIXEL_LIMIT
CAMERA_CONTROL_PROMPT_TRIGGER = camera_warp.CAMERA_CONTROL_PROMPT_TRIGGER
Pi3XWarpRenderer = camera_warp.Pi3XWarpRenderer
Pi3XWarpRendererConfig = camera_warp.Pi3XWarpRendererConfig
center_crop_resize_first_frame = camera_warp.center_crop_resize_first_frame
se3_inverse = camera_warp.se3_inverse


VIDEO_COLUMNS = ("video", "video_url", "url", "video_path", "path")
PROMPT_COLUMNS = ("prompt", "prompts", "caption", "text")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run online Pi3X warp sampling for warp-as-history training batches."
    )
    parser.add_argument("--csv", type=Path, default=Path("data/training/training_data.csv"))
    parser.add_argument("--data_root", type=Path, default=Path("data/training"))
    parser.add_argument("--output_dir", type=Path, default=Path("runs/training_dryrun"))
    parser.add_argument("--video_column", default="")
    parser.add_argument("--prompt_column", default="")
    parser.add_argument("--prompt_trigger", default=CAMERA_CONTROL_PROMPT_TRIGGER)
    parser.add_argument("--limit_rows", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num_frames", type=int, default=33)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument(
        "--max_video_frames",
        type=int,
        default=0,
        help="Maximum decoded frames per direction. Default 0 matches training and decodes all frames.",
    )

    parser.add_argument("--direction", choices=("training", "forward", "reverse"), default="training")
    parser.add_argument("--direction_augmentation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--direction_reverse_probability", type=float, default=0.5)
    parser.add_argument("--chunk_mode", choices=("training", "first", "later"), default="training")
    parser.add_argument("--first_chunk_prob", type=float, default=0.5)
    parser.add_argument("--max_history_frames", type=int, default=19)
    parser.add_argument("--future_keyframe_prob", type=float, default=0.5)
    parser.add_argument("--future_keyframes_min", type=int, default=1)
    parser.add_argument("--future_keyframes_max", type=int, default=2)

    parser.add_argument("--device", default="")
    parser.add_argument("--pi3_pixel_limit", type=int, default=CAMERA_CONTROL_PI3_PIXEL_LIMIT)
    parser.add_argument("--conf_threshold", type=float, default=0.1)
    parser.add_argument("--depth_edge_rtol", type=float, default=0.03)
    parser.add_argument("--mesh_samples_per_axis", type=int, default=4)
    parser.add_argument("--render_mode", default=CAMERA_CONTROL_DEFAULT_WARP_RENDER_MODE)
    parser.add_argument("--target_fill_radius", type=int, default=CAMERA_CONTROL_DEFAULT_WARP_TARGET_FILL_RADIUS)
    parser.add_argument(
        "--target_fill_min_neighbors",
        type=int,
        default=CAMERA_CONTROL_DEFAULT_WARP_TARGET_FILL_MIN_NEIGHBORS,
    )
    parser.add_argument("--mesh_break_mode", default=CAMERA_CONTROL_DEFAULT_MESH_BREAK_MODE)
    parser.add_argument("--mesh_depth_rtol", type=float, default=CAMERA_CONTROL_DEFAULT_MESH_DEPTH_RTOL)
    parser.add_argument("--mesh_normal_tol_deg", type=float, default=CAMERA_CONTROL_DEFAULT_MESH_NORMAL_TOL_DEG)
    parser.add_argument("--invisible_fill", default=CAMERA_CONTROL_DEFAULT_WARP_INVISIBLE_FILL)
    return parser.parse_args()


def is_uri(value: str) -> bool:
    parsed = urlparse(value)
    return bool(parsed.scheme) and parsed.scheme not in {"", "file"}


def resolve_video_ref(value: str, data_root: Path) -> str | Path:
    text = str(value).strip()
    if is_uri(text):
        return text
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = data_root / path
    return path


def pick_column(header: list[str], requested: str, candidates: tuple[str, ...], label: str) -> str:
    if requested:
        if requested not in header:
            raise KeyError(f"Requested {label} column {requested!r} is missing from CSV header {header}.")
        return requested
    for name in candidates:
        if name in header:
            return name
    raise KeyError(f"Could not infer {label} column from CSV header {header}; pass --{label}_column.")


def add_prompt_trigger(prompt: str, trigger: str | None) -> str:
    prompt = str(prompt or "").strip()
    trigger = str(trigger or "").strip()
    if not trigger:
        return prompt
    if prompt.startswith(trigger):
        return prompt
    return f"{trigger} {prompt}".strip()


def stable_seed_from_parts(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(int(base_seed))] + [str(part) for part in parts]).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**63 - 1)


def next_index_generator(num_items: int, max_steps: int, shuffle: bool, seed: int):
    if num_items <= 0:
        raise ValueError("No training items.")
    generator = torch.Generator().manual_seed(int(seed))
    order: list[int] = []
    cursor = 0
    for step in range(int(max_steps)):
        if not shuffle:
            yield step % num_items
            continue
        if cursor >= len(order):
            order = torch.randperm(num_items, generator=generator).tolist()
            cursor = 0
        idx = int(order[cursor])
        cursor += 1
        yield idx


def load_rows(args: argparse.Namespace) -> tuple[list[dict[str, str]], str, str]:
    with args.csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {args.csv}")
        header = list(reader.fieldnames)
        video_column = pick_column(header, args.video_column, VIDEO_COLUMNS, "video")
        prompt_column = pick_column(header, args.prompt_column, PROMPT_COLUMNS, "prompt")
        raw_rows = [dict(row) for row in reader]
    if args.limit_rows > 0:
        raw_rows = raw_rows[: int(args.limit_rows)]
    rows: list[dict[str, str]] = []
    for row_index, row in enumerate(raw_rows):
        base = dict(row)
        raw_prompt = str(base.get(prompt_column, ""))
        base["id"] = str(base.get("id") or f"online_{row_index:06d}")
        base["online_row_index"] = str(row_index)
        base["video_path"] = str(base[video_column])
        base["prompt_raw"] = raw_prompt
        base["prompt"] = add_prompt_trigger(raw_prompt, args.prompt_trigger)
        rows.append(base)
    if not rows:
        raise ValueError(f"CSV has no data rows: {args.csv}")
    return rows, "video_path", "prompt"


def iter_image_files(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def load_video_frames(
    ref: str | Path,
    *,
    height: int,
    width: int,
    frame_stride: int,
    max_video_frames: int,
) -> list[Image.Image]:
    frame_stride = max(1, int(frame_stride))
    max_video_frames = int(max_video_frames)
    frames: list[Image.Image] = []

    if isinstance(ref, Path) and ref.is_dir():
        for src_idx, path in enumerate(iter_image_files(ref)):
            if src_idx % frame_stride != 0:
                continue
            frame = Image.open(path).convert("RGB")
            frames.append(center_crop_resize_first_frame(frame, height, width))
            if max_video_frames > 0 and len(frames) >= max_video_frames:
                break
    else:
        reader = imageio.get_reader(str(ref))
        try:
            for src_idx, array in enumerate(reader):
                if src_idx % frame_stride != 0:
                    continue
                frame = Image.fromarray(np.asarray(array)).convert("RGB")
                frames.append(center_crop_resize_first_frame(frame, height, width))
                if max_video_frames > 0 and len(frames) >= max_video_frames:
                    break
        finally:
            reader.close()

    if not frames:
        raise ValueError(f"No frames decoded from {ref}.")
    return frames


def pil_to_tensor(frame: Image.Image) -> torch.Tensor:
    arr = np.asarray(frame.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return tensor * 2.0 - 1.0


def tensor_video_to_pil_frames(video: torch.Tensor) -> list[Image.Image]:
    if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 3:
        raise ValueError(f"Expected video tensor [1, 3, T, H, W], got {tuple(video.shape)}.")
    arr = video[0].detach().float().cpu().clamp(-1.0, 1.0)
    arr = ((arr + 1.0) * 127.5).round().to(torch.uint8)
    arr = arr.permute(1, 2, 3, 0).numpy()
    return [Image.fromarray(frame, mode="RGB") for frame in arr]


def mask_tensor_to_pil_frames(mask: torch.Tensor) -> list[Image.Image]:
    if mask.ndim != 5 or mask.shape[0] != 1 or mask.shape[1] != 1:
        raise ValueError(f"Expected visibility mask tensor [1, 1, T, H, W], got {tuple(mask.shape)}.")
    arr = mask[0, 0].detach().float().cpu().clamp(0.0, 1.0)
    arr = (arr * 255.0).round().to(torch.uint8).numpy()
    return [Image.fromarray(frame, mode="L") for frame in arr]


def subset_geometry(full_geometry: dict[str, Any], keyframe_indices: list[int]) -> dict[str, Any]:
    if not keyframe_indices:
        raise ValueError("At least one keyframe is required for warp rendering.")
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


def relative_poses(full_geometry: dict[str, Any], source_pose: np.ndarray, target_indices: list[int]) -> np.ndarray:
    keyframe_geometries = full_geometry["keyframe_geometries"]
    target_world = np.stack(
        [np.asarray(keyframe_geometries[int(idx)]["source_pose"], dtype=np.float32) for idx in target_indices],
        axis=0,
    )
    source_inv = se3_inverse(np.asarray(source_pose, dtype=np.float32)[None])[0]
    return np.einsum("ij,tjk->tik", source_inv.astype(np.float32, copy=False), target_world).astype(np.float32)


class PreparedVideoCache:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        rows: list[dict[str, str]],
        video_column: str,
        renderer: Pi3XWarpRenderer,
        device: torch.device,
    ) -> None:
        self.args = args
        self.rows = rows
        self.video_column = video_column
        self.renderer = renderer
        self.device = device
        self.cache: dict[tuple[int, str], dict[str, Any]] = {}

    def get(self, row_index: int, direction: str) -> dict[str, Any]:
        key = (int(row_index), str(direction))
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        row = self.rows[int(row_index)]
        ref = resolve_video_ref(row[self.video_column], self.args.data_root)
        forward_frames = load_video_frames(
            ref,
            height=int(self.args.height),
            width=int(self.args.width),
            frame_stride=int(self.args.frame_stride),
            max_video_frames=int(self.args.max_video_frames),
        )
        frames = forward_frames if direction == "forward" else list(reversed(forward_frames))
        tensors = [pil_to_tensor(frame).unsqueeze(0) for frame in frames]
        print(
            json.dumps(
                {
                    "event": "estimate_geometry",
                    "row_index": int(row_index),
                    "direction": direction,
                    "frames": len(frames),
                    "video": str(ref),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        geometry = self.renderer.estimate_keyframe_geometry(tensors, device=self.device)
        cached = {
            "direction": direction,
            "frames": frames,
            "geometry": geometry,
            "row": row,
            "row_index": int(row_index),
            "video_ref": str(ref),
        }
        self.cache[key] = cached
        return cached


def choose_direction(args: argparse.Namespace, rng: random.Random) -> str:
    if args.direction != "training":
        return str(args.direction)
    if not bool(args.direction_augmentation):
        return "forward"
    reverse_prob = float(args.direction_reverse_probability)
    return "reverse" if rng.random() < reverse_prob else "forward"


def choose_chunk_mode(args: argparse.Namespace, rng: random.Random, *, total_frames: int, num_frames: int) -> str:
    if args.chunk_mode != "training":
        return str(args.chunk_mode)
    return "first" if rng.random() < float(args.first_chunk_prob) or total_frames <= num_frames else "later"


def sample_training_case(
    *,
    prepared: dict[str, Any],
    args: argparse.Namespace,
    rng: random.Random,
    sample_index: int,
    train_step: int,
    prepare_index: int,
) -> dict[str, Any]:
    frames: list[Image.Image] = prepared["frames"]
    n = len(frames)
    num_frames = int(args.num_frames)
    if n < num_frames:
        raise ValueError(f"Need at least {num_frames} decoded frames, got {n} for {prepared['video_ref']}.")

    chunk_mode = choose_chunk_mode(args, rng, total_frames=n, num_frames=num_frames)
    if chunk_mode == "later" and n <= num_frames:
        chunk_mode = "first"

    if chunk_mode == "first":
        source_idx = rng.randint(0, n - num_frames)
        target_indices = list(range(source_idx, source_idx + num_frames))
        history_indices: list[int] = []
        keyframe_indices = [source_idx]
        render_pose_indices = target_indices
        drop_renderer_source = False
        future_keyframe_indices: list[int] = []
        keyframe_policy = "source_only"
        condition_idx = source_idx
        condition_frame = frames[source_idx]
    else:
        target_start = rng.randint(1, n - num_frames)
        target_indices = list(range(target_start, target_start + num_frames))
        max_history = min(int(args.max_history_frames), target_start)
        history_len = rng.randint(1, max(1, max_history))
        history_indices = list(range(target_start - history_len, target_start))
        future_keyframe_indices = []
        keyframe_policy = "history_only"
        if rng.random() < float(args.future_keyframe_prob):
            future_min = max(0, int(args.future_keyframes_min))
            future_max = max(future_min, int(args.future_keyframes_max))
            future_count = rng.randint(future_min, future_max)
            future_count = min(future_count, len(target_indices))
            if future_count > 0:
                future_keyframe_indices = sorted(rng.sample(target_indices, future_count))
            keyframe_policy = "history_plus_future"
        keyframe_indices = sorted(set(history_indices + future_keyframe_indices))
        render_pose_indices = [keyframe_indices[-1], *target_indices]
        drop_renderer_source = True
        condition_idx = history_indices[-1]
        condition_frame = frames[condition_idx]

    geometry = subset_geometry(prepared["geometry"], keyframe_indices)
    poses = relative_poses(prepared["geometry"], geometry["source_pose"], render_pose_indices)
    seq = f"{prepared['row']['id']}:{prepared['direction']}:{chunk_mode}:{prepare_index}"
    return {
        "chunk_mode": chunk_mode,
        "condition_frame": condition_frame,
        "condition_idx": condition_idx,
        "direction": prepared["direction"],
        "drop_renderer_source": drop_renderer_source,
        "future_keyframe_indices": future_keyframe_indices,
        "gt_history_frames": [frames[idx] for idx in history_indices],
        "gt_target_frames": [frames[idx] for idx in target_indices],
        "history_indices": history_indices,
        "keyframe_indices": keyframe_indices,
        "keyframe_policy": keyframe_policy,
        "metadata": {
            "chunk_mode": chunk_mode,
            "condition_idx": int(condition_idx),
            "direction": prepared["direction"],
            "future_keyframe_indices": future_keyframe_indices,
            "history_indices": history_indices,
            "keyframe_indices": keyframe_indices,
            "keyframe_policy": keyframe_policy,
            "prepare_index": prepare_index,
            "prompt": str(prepared["row"].get("prompt", "")),
            "prompt_raw": str(prepared["row"].get("prompt_raw", "")),
            "prompt_trigger": str(args.prompt_trigger or ""),
            "render_pose_indices": render_pose_indices,
            "row_index": prepared["row_index"],
            "row_id": str(prepared["row"]["id"]),
            "sample_index": int(sample_index),
            "seq": seq,
            "target_indices": target_indices,
            "train_step": int(train_step),
            "video": prepared["video_ref"],
        },
        "render_pose_indices": render_pose_indices,
        "target_relative_poses": poses,
        "target_indices": target_indices,
        "warp_geometry": geometry,
    }


def render_warp_video(
    *,
    case: dict[str, Any],
    renderer: Pi3XWarpRenderer,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[Image.Image], list[Image.Image]]:
    rendered = renderer.render_from_geometry(
        case["warp_geometry"],
        case["target_relative_poses"],
        height=int(args.height),
        width=int(args.width),
        device=device,
        invisible_fill_mode=str(args.invisible_fill),
        render_mode=str(args.render_mode),
        target_fill_radius=int(args.target_fill_radius),
        target_fill_min_neighbors=int(args.target_fill_min_neighbors),
        mesh_break_mode=str(args.mesh_break_mode),
    )
    warp_video = rendered["warp_video"]
    warp_mask = rendered["warp_visibility_mask"]
    if case["drop_renderer_source"]:
        warp_video = warp_video[:, :, 1:]
        warp_mask = warp_mask[:, :, 1:]
    warp_frames = tensor_video_to_pil_frames(warp_video)
    mask_frames = mask_tensor_to_pil_frames(warp_mask)
    expected = len(case["gt_target_frames"])
    if len(warp_frames) != expected or len(mask_frames) != expected:
        raise ValueError(f"Rendered {len(warp_frames)} warp frames/{len(mask_frames)} masks, expected {expected}.")
    case["metadata"]["warp_render_stats"] = rendered.get("warp_render_stats", {})
    case["metadata"]["mesh_break_stats"] = rendered.get("mesh_break_stats", [])
    return warp_frames, mask_frames


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: tuple[int, int, int]) -> None:
    draw.rectangle((xy[0] - 3, xy[1] - 2, xy[0] + min(len(text) * 7 + 6, 780), xy[1] + 14), fill=(0, 0, 0))
    draw.text(xy, text, fill=fill)


def compose_preview_frames(
    *,
    case: dict[str, Any],
    warp_frames: list[Image.Image],
    mask_frames: list[Image.Image],
    width: int,
    height: int,
) -> list[np.ndarray]:
    target_frames = case["gt_target_frames"]
    canvas_h = height * 4
    out: list[np.ndarray] = []
    for j, (target_frame, warp_frame, mask_frame) in enumerate(zip(target_frames, warp_frames, mask_frames)):
        canvas = Image.new("RGB", (width, canvas_h), (0, 0, 0))
        canvas.paste(case["condition_frame"].resize((width, height), Image.Resampling.BILINEAR), (0, 0))
        canvas.paste(target_frame.resize((width, height), Image.Resampling.BILINEAR), (0, height))
        canvas.paste(warp_frame.resize((width, height), Image.Resampling.BILINEAR), (0, height * 2))
        mask_rgb = Image.merge("RGB", [mask_frame, mask_frame, mask_frame])
        canvas.paste(mask_rgb.resize((width, height), Image.Resampling.NEAREST), (0, height * 3))
        draw = ImageDraw.Draw(canvas)
        for y in (height, height * 2, height * 3):
            draw.rectangle((0, y - 2, width, y + 2), fill=(255, 255, 255))
        meta = case["metadata"]
        title = (
            f"step={meta.get('train_step', meta['sample_index'])} seq={meta['seq']} "
            f"{meta['keyframe_policy']} target_idx={case['target_indices'][j]}"
        )
        draw_label(draw, (6, 5), f"condition image idx={meta['condition_idx']}", (255, 255, 255))
        draw_label(draw, (6, height + 5), "target latents source: GT target frame", (255, 255, 255))
        draw_label(draw, (6, height * 2 + 5), "history video input: online warp frame", (255, 255, 255))
        draw_label(draw, (6, height * 3 + 5), "history visible mask used by training", (255, 255, 255))
        draw_label(draw, (6, height - 22), title, (128, 255, 128))
        out.append(np.asarray(canvas, dtype=np.uint8))
    return out


def write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=int(fps), codec="libx264", macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.prompt_column = str(args.prompt_column)

    rows, video_column, prompt_column = load_rows(args)
    args.prompt_column = prompt_column
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    renderer = Pi3XWarpRenderer(
        Pi3XWarpRendererConfig(
            pi3_pixel_limit=int(args.pi3_pixel_limit),
            conf_threshold=float(args.conf_threshold),
            depth_edge_rtol=float(args.depth_edge_rtol),
            mesh_samples_per_axis=int(args.mesh_samples_per_axis),
            render_mode=str(args.render_mode),
            target_fill_radius=int(args.target_fill_radius),
            target_fill_min_neighbors=int(args.target_fill_min_neighbors),
            mesh_break_mode=str(args.mesh_break_mode),
            mesh_depth_rtol=float(args.mesh_depth_rtol),
            mesh_normal_tol_deg=float(args.mesh_normal_tol_deg),
        )
    )
    cache = PreparedVideoCache(
        args=args,
        rows=rows,
        video_column=video_column,
        renderer=renderer,
        device=device,
    )

    videos_dir = args.output_dir / "videos"
    manifest_path = args.output_dir / "manifest.jsonl"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    batch_frames: list[np.ndarray] = []
    total_steps = int(args.start_step) + int(args.num_samples)
    index_iter = next_index_generator(len(rows), total_steps, bool(args.shuffle), int(args.seed))
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for train_step in range(total_steps):
            row_index = next(index_iter)
            if train_step < int(args.start_step):
                continue
            sample_index = train_step - int(args.start_step)
            row = rows[row_index]
            prepare_index = int(train_step) + 1
            rng = random.Random(
                stable_seed_from_parts(int(args.seed), "online_warp_training", row["id"], int(prepare_index))
            )
            direction = choose_direction(args, rng)
            prepared = cache.get(row_index, direction)
            case = sample_training_case(
                prepared=prepared,
                args=args,
                rng=rng,
                sample_index=sample_index,
                train_step=train_step,
                prepare_index=prepare_index,
            )
            warp_frames, mask_frames = render_warp_video(case=case, renderer=renderer, args=args, device=device)
            preview_frames = compose_preview_frames(
                case=case,
                warp_frames=warp_frames,
                mask_frames=mask_frames,
                width=int(args.width),
                height=int(args.height),
            )
            sample_path = videos_dir / f"sample_{sample_index:03d}_{case['chunk_mode']}_{direction}.mp4"
            write_video(sample_path, preview_frames, fps=int(args.fps))
            case["metadata"]["preview_video"] = str(sample_path)
            manifest.write(json.dumps(case["metadata"], ensure_ascii=False) + "\n")
            manifest.flush()
            batch_frames.extend(preview_frames)
            gap = np.zeros_like(preview_frames[-1])
            batch_frames.extend([gap] * max(1, int(args.fps // 2)))
            print(json.dumps({"event": "wrote_preview", **case["metadata"]}, ensure_ascii=False), flush=True)

    if batch_frames:
        write_video(args.output_dir / "batch_preview.mp4", batch_frames, fps=int(args.fps))
    print(
        json.dumps(
            {
                "event": "done",
                "batch_preview": str(args.output_dir / "batch_preview.mp4"),
                "manifest": str(manifest_path),
                "video_column": video_column,
                "prompt_column": prompt_column,
                "seed": int(args.seed),
                "shuffle": bool(args.shuffle),
                "start_step": int(args.start_step),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
