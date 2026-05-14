from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from safetensors.torch import load_file, save_file

from helios.diffusers_version.pipeline_helios_diffusers import (
    XLA_AVAILABLE,
    HeliosPipeline,
    calculate_shift,
    optimized_scale,
)

from .camera_warp import (
    CAMERA_CONTROL_DEFAULT_MESH_BREAK_MODE,
    CAMERA_CONTROL_DEFAULT_MESH_DEPTH_RTOL,
    CAMERA_CONTROL_DEFAULT_MESH_NORMAL_TOL_DEG,
    CAMERA_CONTROL_DEFAULT_PI3X_KEYFRAME_MEMORY,
    CAMERA_CONTROL_DEFAULT_TRANSLATION_SCALE,
    CAMERA_CONTROL_DEFAULT_TRANSLATION_SCALE_USE_FIRST_FRAME_DEPTH,
    CAMERA_CONTROL_DEFAULT_WARP_INVISIBLE_FILL,
    CAMERA_CONTROL_DEFAULT_WARP_RENDER_MODE,
    CAMERA_CONTROL_DEFAULT_WARP_TARGET_FILL_MIN_NEIGHBORS,
    CAMERA_CONTROL_DEFAULT_WARP_TARGET_FILL_RADIUS,
    CAMERA_CONTROL_PROMPT_TRIGGER,
    Pi3XWarpRenderer,
    Pi3XWarpRendererConfig,
    center_crop_resize_first_frame,
)
from .defaults import (
    LORA_DISABLED_VALUES,
    WAH_HISTORY_SIZES,
    WAH_INVISIBLE_FILL_MODES,
    WAH_NEGATIVE_PROMPT,
    WAH_NUM_FRAMES,
    WAH_NUM_LATENT_FRAMES_PER_CHUNK,
    WAH_PREV_CHUNK_HISTORY_SIZES,
    WAH_PROMPT_TRIGGER,
    WAH_PYRAMID_NUM_STAGES,
    WAH_PYRAMID_STEPS,
    WAH_VISIBLE_TOKEN_THRESHOLD,
)


if XLA_AVAILABLE:
    import torch_xla.core.xla_model as xm


def _normalize_optional_lora_path(lora_path: str | Path | None) -> str | None:
    if lora_path is None:
        return None
    lora_path_str = str(lora_path)
    if lora_path_str.strip().lower() in LORA_DISABLED_VALUES:
        return None
    return lora_path_str


def _optional_to_dtype(tensor: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.to(dtype=dtype)


def _display_boundary_frame(frame: torch.Tensor) -> torch.Tensor:
    frame01 = (frame.detach().float().cpu() / 2.0 + 0.5).clamp(0.0, 1.0)
    frame01 = (frame01 * 255.0).round() / 255.0
    return frame01 * 2.0 - 1.0


def _normalize_history_sizes(history_sizes: list[int] | tuple[int, int, int], name: str) -> tuple[int, int, int]:
    normalized = tuple(int(x) for x in history_sizes)
    if len(normalized) != 3 or any(x < 0 for x in normalized):
        raise ValueError(f"{name} must contain exactly three non-negative integers, got {history_sizes!r}.")
    return normalized


def _checkpoint_model_path(value: str | Path, *, label: str) -> str:
    repo_root = Path(__file__).resolve().parents[1]
    checkpoints_root = Path(str((repo_root / "checkpoints").absolute()))
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    path = Path(str(path.absolute()))
    if not path.is_relative_to(checkpoints_root):
        raise ValueError(f"{label} must be under {checkpoints_root}, got {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {label} directory: {path}. Run `python scripts/check_models.py`.")
    return str(path)


class WarpAsHistoryPipeline(HeliosPipeline):
    """Minimal Warp-as-History inference pipeline.

    Pass `warp_video` when geometry preprocessing was done externally. If
    `warp_video` is omitted, pass `camera_poses` and the pipeline will estimate
    first-frame Pi3X geometry and render the warp condition internally.
    """

    _wah_adapter_name = "wah"
    _camera_warp_renderer: Pi3XWarpRenderer | None = None
    _camera_warp_renderer_key: tuple[Any, ...] | None = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, *args: Any, **kwargs: Any):
        model_path = _checkpoint_model_path(
            pretrained_model_name_or_path,
            label="pretrained_model_name_or_path",
        )
        return super().from_pretrained(model_path, *args, **kwargs)

    @staticmethod
    def _add_prompt_trigger(
        prompt: str | list[str] | None,
        trigger: str | None = WAH_PROMPT_TRIGGER,
    ) -> str | list[str] | None:
        trigger_text = "" if trigger is None else str(trigger).strip()
        if not trigger_text:
            return prompt

        def add_one(text: str) -> str:
            stripped = text.lstrip()
            if stripped.startswith(trigger_text):
                return text
            return f"{trigger_text} {text}"

        if prompt is None:
            return None
        if isinstance(prompt, list):
            return [add_one(item) for item in prompt]
        return add_one(prompt)

    @staticmethod
    def _coerce_visibility_mask(mask: Any) -> torch.Tensor | None:
        if mask is None:
            return None
        if torch.is_tensor(mask):
            tensor = mask.detach()
        elif isinstance(mask, np.ndarray):
            tensor = torch.from_numpy(mask)
        elif isinstance(mask, (list, tuple)):
            if len(mask) == 0:
                raise ValueError("warp_visibility_mask cannot be an empty list.")
            if all(torch.is_tensor(item) for item in mask):
                tensor = torch.stack([item.detach() for item in mask], dim=0)
            else:
                tensor = torch.from_numpy(np.stack([np.asarray(item) for item in mask], axis=0))
        else:
            tensor = torch.from_numpy(np.asarray(mask))

        tensor = tensor.float()
        if tensor.ndim == 5:
            if tensor.shape[1] in {1, 3, 4}:
                pass
            elif tensor.shape[-1] in {1, 3, 4}:
                tensor = tensor.permute(0, 4, 1, 2, 3)
            else:
                raise ValueError(
                    "warp_visibility_mask must be [B,1,T,H,W], [B,T,H,W,1], or a compatible mask tensor."
                )
        elif tensor.ndim == 4:
            if tensor.shape[-1] in {1, 3, 4}:
                tensor = tensor.permute(3, 0, 1, 2).unsqueeze(0)
            else:
                tensor = tensor.unsqueeze(1)
        elif tensor.ndim == 3:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError(f"warp_visibility_mask must be 2D-5D, got shape {tuple(tensor.shape)}.")

        if tensor.shape[1] != 1:
            tensor = tensor[:, :3].mean(dim=1, keepdim=True)
        if tensor.numel() and float(tensor.max()) > 1.0:
            tensor = tensor / 255.0
        return tensor.clamp(0.0, 1.0)

    @staticmethod
    def _resize_visibility_mask(
        mask: torch.Tensor,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = mask.to(device=device, dtype=torch.float32)
        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1, -1, -1, -1)
        if mask.shape[0] != batch_size:
            raise ValueError(f"warp_visibility_mask batch size must be 1 or {batch_size}, got {mask.shape[0]}.")
        if mask.shape[2:] != (num_frames, height, width):
            mask = F.interpolate(mask, size=(num_frames, height, width), mode="trilinear", align_corners=False)
        return mask.clamp(0.0, 1.0)

    @staticmethod
    def _coerce_warp_video_tensor(warp_video: Any, height: int, width: int, device: torch.device) -> torch.Tensor:
        if isinstance(warp_video, (list, tuple)):
            if len(warp_video) == 0:
                raise ValueError("warp_video cannot be an empty list.")
            frames = []
            for frame in warp_video:
                if isinstance(frame, Image.Image):
                    image = frame.convert("RGB")
                else:
                    array = np.asarray(frame)
                    if array.ndim == 2:
                        array = np.repeat(array[..., None], 3, axis=-1)
                    array = array[..., :3]
                    if array.dtype != np.uint8:
                        array = array.astype(np.float32, copy=False)
                        if array.size and float(np.nanmax(array)) <= 1.5:
                            array = array * 255.0
                        array = np.clip(array, 0.0, 255.0).round().astype(np.uint8)
                    image = Image.fromarray(array).convert("RGB")
                if image.size != (int(width), int(height)):
                    image = image.resize((int(width), int(height)), Image.Resampling.BILINEAR)
                frames.append(np.asarray(image, dtype=np.uint8))
            video = torch.from_numpy(np.stack(frames, axis=0)).float() / 255.0
            video = video.permute(0, 3, 1, 2)
            video = F.interpolate(video, size=(int(height), int(width)), mode="bilinear", align_corners=False)
            tensor = video.unsqueeze(0).permute(0, 2, 1, 3, 4)
            return (tensor * 2.0 - 1.0).to(device=device, dtype=torch.float32).clamp(-1.0, 1.0)
        elif isinstance(warp_video, np.ndarray):
            tensor = torch.from_numpy(warp_video)
        elif torch.is_tensor(warp_video):
            tensor = warp_video.detach()
        else:
            raise ValueError(
                "warp_video must be a tensor, numpy array, or a list of PIL/numpy/torch image frames."
            )

        tensor = tensor.float()
        if tensor.ndim == 5:
            if tensor.shape[1] in {1, 3, 4}:  # [B, C, T, H, W]
                pass
            elif tensor.shape[2] in {1, 3, 4}:  # [B, T, C, H, W]
                tensor = tensor.permute(0, 2, 1, 3, 4)
            elif tensor.shape[-1] in {1, 3, 4}:  # [B, T, H, W, C]
                tensor = tensor.permute(0, 4, 1, 2, 3)
            else:
                raise ValueError(f"Unsupported 5D warp_video shape: {tuple(tensor.shape)}.")
        elif tensor.ndim == 4:
            if tensor.shape[0] in {1, 3, 4}:  # [C, T, H, W]
                tensor = tensor.unsqueeze(0)
            elif tensor.shape[1] in {1, 3, 4}:  # [T, C, H, W]
                tensor = tensor.permute(1, 0, 2, 3).unsqueeze(0)
            elif tensor.shape[-1] in {1, 3, 4}:  # [T, H, W, C]
                tensor = tensor.permute(3, 0, 1, 2).unsqueeze(0)
            else:
                raise ValueError(f"Unsupported 4D warp_video shape: {tuple(tensor.shape)}.")
        else:
            raise ValueError(f"warp_video must be 4D or 5D after conversion, got {tuple(tensor.shape)}.")

        if tensor.shape[1] == 1:
            tensor = tensor.expand(-1, 3, -1, -1, -1)
        elif tensor.shape[1] > 3:
            tensor = tensor[:, :3]

        if tensor.numel():
            if float(tensor.max()) > 2.0:
                tensor = tensor / 255.0
            if float(tensor.min()) >= 0.0:
                tensor = tensor * 2.0 - 1.0

        if tensor.shape[-2:] != (int(height), int(width)):
            batch_size, channels, num_frames, old_height, old_width = tensor.shape
            tensor = tensor.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, channels, old_height, old_width)
            tensor = F.interpolate(tensor, size=(int(height), int(width)), mode="bilinear", align_corners=False)
            tensor = tensor.reshape(batch_size, num_frames, channels, int(height), int(width)).permute(0, 2, 1, 3, 4)

        return tensor.to(device=device, dtype=torch.float32).clamp(-1.0, 1.0)

    @staticmethod
    def _visibility_mask_to_history_latents(
        mask: torch.Tensor,
        latent_frames: int,
        latent_height: int,
        latent_width: int,
        temporal_scale: int,
    ) -> torch.Tensor:
        sample_ids = torch.arange(int(latent_frames), dtype=torch.long, device=mask.device) * int(
            max(1, temporal_scale)
        )
        sample_ids = sample_ids.clamp(max=mask.shape[2] - 1)
        sampled = mask.index_select(2, sample_ids)
        sampled = F.interpolate(
            sampled,
            size=(int(latent_frames), int(latent_height), int(latent_width)),
            mode="trilinear",
            align_corners=False,
        )
        return sampled.clamp(0.0, 1.0)

    def _latent_stats(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(device, self.vae.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(device, self.vae.dtype)
        return latents_mean, latents_std

    @staticmethod
    def _slice_frame_sequence(
        value: torch.Tensor | np.ndarray | None,
        start: int,
        frame_count: int,
        name: str,
    ) -> torch.Tensor | np.ndarray | None:
        if value is None:
            return None
        start = max(0, int(start))
        frame_count = int(frame_count)
        if frame_count <= 0:
            raise ValueError(f"{name} frame_count must be positive, got {frame_count}.")
        if torch.is_tensor(value):
            if value.shape[0] <= 0:
                raise ValueError(f"{name} must contain at least one frame.")
            start = min(start, int(value.shape[0]) - 1)
            sliced = value.detach()[start : start + frame_count]
            if sliced.shape[0] < frame_count:
                pad = sliced[-1:].repeat(frame_count - int(sliced.shape[0]), *([1] * (sliced.ndim - 1)))
                sliced = torch.cat([sliced, pad], dim=0)
            return sliced

        array = np.asarray(value)
        if array.shape[0] <= 0:
            raise ValueError(f"{name} must contain at least one frame.")
        start = min(start, int(array.shape[0]) - 1)
        sliced = array[start : start + frame_count]
        if sliced.shape[0] < frame_count:
            pad = np.repeat(sliced[-1:], frame_count - int(sliced.shape[0]), axis=0)
            sliced = np.concatenate([sliced, pad], axis=0)
        return sliced

    def _prepare_warp_state(
        self,
        image: PipelineImageInput,
        warp_video: Any | None,
        warp_visibility_mask: Any,
        height: int,
        width: int,
        num_frames: int,
        warp_invisible_fill: str,
        visible_token_drop: bool,
        rope_alignment: bool,
        prev_chunk_history_sizes: list[int] | tuple[int, int, int],
        add_noise_to_warp_latents: bool,
        warp_noise_sigma_min: float,
        warp_noise_sigma_max: float,
        image_history_prefix_noised: bool,
        generator: torch.Generator | list[torch.Generator] | None,
        lora_active: bool,
    ) -> dict[str, Any]:
        if warp_invisible_fill not in WAH_INVISIBLE_FILL_MODES:
            raise ValueError(f"warp_invisible_fill must be one of {sorted(WAH_INVISIBLE_FILL_MODES)}.")

        device = self._execution_device
        window_num_frames = (WAH_NUM_LATENT_FRAMES_PER_CHUNK - 1) * self.vae_scale_factor_temporal + 1
        num_warp_chunks = max(1, (max(int(num_frames), 1) + window_num_frames - 1) // window_num_frames)
        total_warp_frames = num_warp_chunks * window_num_frames
        prev_chunk_history_sizes = _normalize_history_sizes(prev_chunk_history_sizes, "prev_chunk_history_sizes")
        history_capacity = int(sum(WAH_HISTORY_SIZES))
        if sum(prev_chunk_history_sizes) > history_capacity:
            raise ValueError(
                "sum(prev_chunk_history_sizes) must be <= "
                f"{history_capacity}, got {prev_chunk_history_sizes!r}."
            )
        if any(size > cap for size, cap in zip(prev_chunk_history_sizes, WAH_HISTORY_SIZES)):
            raise ValueError(
                "prev_chunk_history_sizes cannot exceed the official WAH history slots "
                f"{WAH_HISTORY_SIZES!r}, got {prev_chunk_history_sizes!r}."
            )
        warp_noise_sigma_min = float(warp_noise_sigma_min)
        warp_noise_sigma_max = float(warp_noise_sigma_max)
        if warp_noise_sigma_min < 0.0 or warp_noise_sigma_max < warp_noise_sigma_min:
            raise ValueError(
                "warp noise sigma range must satisfy 0 <= min <= max, got "
                f"[{warp_noise_sigma_min}, {warp_noise_sigma_max}]."
            )

        source_image = self.video_processor.preprocess(image, height=height, width=width).to(
            device=device, dtype=torch.float32
        )
        warp_video_tensor = None
        visibility_mask = None
        if source_image.shape[0] != 1:
            raise ValueError("WarpAsHistoryPipeline currently supports batch size 1.")
        if warp_video is not None:
            warp_video_tensor = self._coerce_warp_video_tensor(warp_video, height=height, width=width, device=device)
            if warp_video_tensor.shape[0] != 1:
                raise ValueError("WarpAsHistoryPipeline currently supports batch size 1.")
            if warp_video_tensor.shape[2] != total_warp_frames:
                raise ValueError(
                    "warp_video must contain exactly one full warp rollout for the requested frame count: "
                    f"{total_warp_frames} frames for num_frames={int(num_frames)} "
                    f"({num_warp_chunks} chunks x {window_num_frames} frames)."
                )

            visibility_mask = self._coerce_visibility_mask(warp_visibility_mask)
            if visibility_mask is None:
                visibility_mask = torch.ones(
                    1,
                    1,
                    total_warp_frames,
                    height,
                    width,
                    device=device,
                    dtype=torch.float32,
                )
            else:
                visibility_mask = self._resize_visibility_mask(
                    visibility_mask,
                    batch_size=warp_video_tensor.shape[0],
                    num_frames=total_warp_frames,
                    height=height,
                    width=width,
                    device=device,
                )

        return {
            "chunk_index": 0,
            "height": int(height),
            "num_warp_chunks": num_warp_chunks,
            "prev_chunk_history_sizes": prev_chunk_history_sizes,
            "prev_history_latent_window": None,
            "prev_chunk_last_frame": None,
            "warp_history_prefix_latent": None,
            "warp_latents_tensor": None,
            "source_image": source_image,
            "visibility_mask": visibility_mask,
            "warp_invisible_fill": str(warp_invisible_fill),
            "warp_video_tensor": warp_video_tensor,
            "width": int(width),
            "window_num_frames": window_num_frames,
            "visible_token_drop": bool(visible_token_drop),
            "rope_alignment": bool(rope_alignment),
            "add_noise_to_warp_latents": bool(add_noise_to_warp_latents),
            "warp_noise_sigma_min": warp_noise_sigma_min,
            "warp_noise_sigma_max": warp_noise_sigma_max,
            "image_history_prefix_noised": bool(image_history_prefix_noised),
            "lora_active": bool(lora_active),
            "using_camera_warp": False,
        }

    def _render_camera_warp_chunk(
        self,
        state: dict[str, Any],
        chunk_index: int,
        source_frame: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rendered = state.get("camera_first_chunk_rendered") if int(chunk_index) == 0 else None
        if rendered is None:
            window_num_frames = int(state["window_num_frames"])
            if int(chunk_index) == 0:
                source_pose_index = 0
                frame_count = window_num_frames
                translation_scale = float(state["camera_control_translation_scale"])
                use_depth_scale = bool(state["camera_control_translation_scale_use_first_frame_depth"])
            else:
                source_pose_index = int(chunk_index) * window_num_frames - 1
                frame_count = window_num_frames + 1
                translation_scale = float(
                    state.get("camera_translation_effective_scale", state["camera_control_translation_scale"])
                )
                use_depth_scale = False

            camera_poses = self._slice_frame_sequence(
                state["camera_poses"],
                source_pose_index,
                frame_count,
                "camera_poses",
            )
            target_intrinsics = self._slice_frame_sequence(
                state.get("target_intrinsics"),
                source_pose_index,
                frame_count,
                "target_intrinsics",
            )
            geometry = None
            pi3x_keyframe_images = state.get("pi3x_keyframe_images")
            if (
                int(chunk_index) > 0
                and pi3x_keyframe_images is not None
                and len(pi3x_keyframe_images) > 1
            ):
                geometry = state["camera_renderer"].estimate_keyframe_geometry(
                    image_tensors=list(pi3x_keyframe_images),
                    device=device,
                    scale_reference_geometry=state.get("camera_first_frame_geometry"),
                )
                state.setdefault("pi3x_keyframe_counts", []).append(int(geometry["keyframe_count"]))
                state.setdefault("pi3x_keyframe_scale_alignment_stats", []).append(
                    geometry.get("scale_alignment_stats", {})
                )
                state.setdefault("pi3x_keyframe_intrinsic_smoothing_stats", []).append(
                    geometry.get("intrinsic_smoothing_stats", {})
                )
            rendered = state["camera_renderer"].render(
                image_tensor=source_frame.to(device=device, dtype=torch.float32),
                camera_poses=camera_poses,
                height=int(state["height"]),
                width=int(state["width"]),
                num_frames=frame_count,
                device=device,
                geometry=geometry,
                target_intrinsics=target_intrinsics,
                chunk_index=int(chunk_index),
                translation_scale=translation_scale,
                translation_scale_use_first_frame_depth=use_depth_scale,
                invisible_fill_mode=str(state["camera_control_warp_invisible_fill"]),
                render_mode=str(state["camera_control_warp_render_mode"]),
                target_fill_radius=int(state["camera_control_warp_target_fill_radius"]),
                target_fill_min_neighbors=int(state["camera_control_warp_target_fill_min_neighbors"]),
                mesh_break_mode=str(state["camera_control_mesh_break_mode"]),
            )

        if int(chunk_index) == 0:
            state["camera_translation_effective_scale"] = float(
                rendered.get("camera_translation_effective_scale", state["camera_control_translation_scale"])
            )
        state.setdefault("camera_warp_chunks", {})[int(chunk_index)] = rendered
        self._last_camera_warp = rendered

        warp_video = rendered["warp_video"].to(device=device, dtype=torch.float32)
        visibility_mask = rendered.get("warp_visibility_mask")
        if visibility_mask is None:
            visibility_mask = self._coerce_visibility_mask(rendered["visibility_frames"])
        visibility_mask = visibility_mask.to(device=device, dtype=torch.float32)
        return warp_video, visibility_mask

    def _check_minimal_inputs(
        self,
        prompt: str | list[str] | None,
        negative_prompt: str | list[str] | None,
        prompt_embeds: torch.Tensor | None,
        negative_prompt_embeds: torch.Tensor | None,
        num_videos_per_prompt: int,
    ) -> None:
        if num_videos_per_prompt != 1:
            raise ValueError("WarpAsHistoryPipeline requires num_videos_per_prompt == 1.")
        if isinstance(prompt, list) and len(prompt) != 1:
            raise ValueError("WarpAsHistoryPipeline currently supports a single prompt.")
        if isinstance(negative_prompt, list) and len(negative_prompt) != 1:
            raise ValueError("WarpAsHistoryPipeline currently supports a single negative prompt.")
        if prompt_embeds is not None and prompt_embeds.shape[0] != 1:
            raise ValueError("WarpAsHistoryPipeline currently supports prompt_embeds batch size 1.")
        if negative_prompt_embeds is not None and negative_prompt_embeds.shape[0] != 1:
            raise ValueError("WarpAsHistoryPipeline currently supports negative_prompt_embeds batch size 1.")

    def _wah_has_loaded_adapters(self) -> bool:
        transformer = getattr(self, "transformer", None)
        peft_config = getattr(transformer, "peft_config", None)
        if isinstance(peft_config, dict) and len(peft_config) > 0:
            return True
        return getattr(self, "_wah_loaded_lora_path", None) is not None

    def _delete_wah_adapter(self) -> None:
        for module in (self, getattr(self, "transformer", None)):
            if module is not None and hasattr(module, "delete_adapters"):
                try:
                    module.delete_adapters(self._wah_adapter_name)
                    return
                except Exception:
                    pass

    def _set_wah_lora_enabled(self, enabled: bool) -> None:
        transformer = getattr(self, "transformer", None)
        if transformer is None or not self._wah_has_loaded_adapters():
            return
        if enabled and hasattr(transformer, "enable_adapters"):
            transformer.enable_adapters()
        elif not enabled and hasattr(transformer, "disable_adapters"):
            transformer.disable_adapters()

    def _configure_wah_lora(self, lora_path: str | None) -> bool:
        if lora_path is None:
            self._set_wah_lora_enabled(False)
            return False

        adapter_path = self._materialize_wah_lora_path(lora_path, self._wah_adapter_name)
        loaded_path = getattr(self, "_wah_loaded_lora_path", None)
        if loaded_path != lora_path:
            if loaded_path is not None:
                self._delete_wah_adapter()
            self.load_lora_weights(adapter_path, adapter_name=self._wah_adapter_name)
            self._wah_loaded_lora_path = lora_path

        if hasattr(self, "set_adapters"):
            self.set_adapters([self._wah_adapter_name], adapter_weights=[1.0])
        self._set_wah_lora_enabled(True)
        return True

    def _get_wah_runtime_lora_temp_dir(self) -> Path:
        temp_dir = getattr(self, "_wah_runtime_lora_temp_dir", None)
        if temp_dir is None:
            temp_dir = tempfile.TemporaryDirectory(prefix="wah_runtime_lora_")
            self._wah_runtime_lora_temp_dir = temp_dir
        return Path(temp_dir.name)

    def _materialize_wah_lora_path(self, lora_path: str | Path, adapter_name: str) -> str:
        path = Path(lora_path).expanduser().resolve()
        if path.is_dir():
            return str(path)

        safe_adapter_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in adapter_name)
        if path.suffix == ".safetensors":
            peft_state = load_file(str(path))
            if any(key.startswith("transformer.") for key in peft_state):
                return str(path)
            materialized = self._get_wah_runtime_lora_temp_dir() / f"materialized_prefixed_{safe_adapter_name}.safetensors"
            save_file(
                {f"transformer.{key}": value.detach().contiguous().cpu() for key, value in peft_state.items()},
                str(materialized),
            )
            return str(materialized)

        loaded = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(loaded, dict):
            raise TypeError(f"Unsupported LoRA checkpoint type from {path}: {type(loaded)}")
        peft_state = loaded.get("peft", loaded)
        if not isinstance(peft_state, dict) or not peft_state:
            raise ValueError(f"No PEFT LoRA tensors found in {path}")
        if not all(torch.is_tensor(value) for value in peft_state.values()):
            raise TypeError(f"LoRA checkpoint {path} contains non-tensor entries in PEFT state.")
        if not any(key.startswith("transformer.") for key in peft_state):
            peft_state = {f"transformer.{key}": value for key, value in peft_state.items()}

        materialized = self._get_wah_runtime_lora_temp_dir() / f"materialized_{safe_adapter_name}.safetensors"
        save_file(
            {key: value.detach().contiguous().cpu() for key, value in peft_state.items()},
            str(materialized),
        )
        return str(materialized)

    def _get_camera_warp_renderer(
        self,
        *,
        camera_control_warp_render_mode: str,
        camera_control_warp_target_fill_radius: int,
        camera_control_warp_target_fill_min_neighbors: int,
        camera_control_mesh_break_mode: str,
        camera_control_mesh_depth_rtol: float,
        camera_control_mesh_normal_tol_deg: float,
    ) -> Pi3XWarpRenderer:
        key = (
            str(camera_control_warp_render_mode),
            int(camera_control_warp_target_fill_radius),
            int(camera_control_warp_target_fill_min_neighbors),
            str(camera_control_mesh_break_mode),
            float(camera_control_mesh_depth_rtol),
            float(camera_control_mesh_normal_tol_deg),
        )
        if self._camera_warp_renderer is None or self._camera_warp_renderer_key != key:
            config = Pi3XWarpRendererConfig(
                render_mode=str(camera_control_warp_render_mode),
                target_fill_radius=int(camera_control_warp_target_fill_radius),
                target_fill_min_neighbors=int(camera_control_warp_target_fill_min_neighbors),
                mesh_break_mode=str(camera_control_mesh_break_mode),
                mesh_depth_rtol=float(camera_control_mesh_depth_rtol),
                mesh_normal_tol_deg=float(camera_control_mesh_normal_tol_deg),
            )
            self._camera_warp_renderer = Pi3XWarpRenderer(config=config)
            self._camera_warp_renderer_key = key
        return self._camera_warp_renderer

    def _wah_execution_device(self) -> torch.device:
        device = getattr(self, "_execution_device", None)
        if device is None:
            device = getattr(self, "device", None)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _add_noise_to_warp_history_latents(
        self,
        state: dict[str, Any],
        warp_latents: torch.Tensor,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None,
    ) -> torch.Tensor:
        if not bool(state.get("add_noise_to_warp_latents", True)):
            return warp_latents
        noise_sigma_min = float(state.get("warp_noise_sigma_min", 0.111))
        noise_sigma_max = float(state.get("warp_noise_sigma_max", 0.135))
        rand_generator = generator[0] if isinstance(generator, list) else generator
        noisy_chunks = []
        chunk_size = int(WAH_NUM_LATENT_FRAMES_PER_CHUNK)
        for chunk_start in range(0, int(warp_latents.shape[2]), chunk_size):
            latent_chunk = warp_latents[:, :, chunk_start : chunk_start + chunk_size]
            chunk_frames = int(latent_chunk.shape[2])
            frame_sigmas = (
                torch.rand(chunk_frames, device=device, generator=rand_generator)
                * (noise_sigma_max - noise_sigma_min)
                + noise_sigma_min
            ).to(dtype=latent_chunk.dtype)
            frame_sigmas = frame_sigmas.view(1, 1, chunk_frames, 1, 1)
            noisy_chunks.append(
                frame_sigmas
                * randn_tensor(latent_chunk.shape, generator=generator, device=device, dtype=latent_chunk.dtype)
                + (1 - frame_sigmas) * latent_chunk
            )
        return torch.cat(noisy_chunks, dim=2)

    def _prepare_external_warp_latents(
        self,
        state: dict[str, Any],
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None,
    ) -> torch.Tensor:
        cached = state.get("warp_latents_tensor")
        if cached is not None:
            return cached.to(device=device, dtype=torch.float32)
        warp_video_tensor = state["warp_video_tensor"]
        if warp_video_tensor is None:
            raise RuntimeError("warp_as_history rollout is missing external warp video.")
        latents_mean, latents_std = self._latent_stats(device)
        _, warp_latents_tensor = self.prepare_video_latents(
            warp_video_tensor.to(device=device, dtype=torch.float32),
            latents_mean=latents_mean,
            latents_std=latents_std,
            num_latent_frames_per_chunk=WAH_NUM_LATENT_FRAMES_PER_CHUNK,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        warp_latents_tensor = self._add_noise_to_warp_history_latents(
            state=state,
            warp_latents=warp_latents_tensor,
            device=device,
            generator=generator,
        )
        state["warp_latents_tensor"] = warp_latents_tensor.detach()
        return warp_latents_tensor

    def _build_pyramid_base_histories(
        self,
        state: dict[str, Any],
        device: torch.device,
        history_dtype: torch.dtype,
        generator: torch.Generator | list[torch.Generator] | None,
        base_latents_history_short: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        chunk_index = int(state.get("chunk_index", 0))
        frame_start = chunk_index * int(state["window_num_frames"])
        frame_end = frame_start + int(state["window_num_frames"])
        if chunk_index == 0:
            source_frame = state["source_image"].to(device=device, dtype=torch.float32)
        else:
            source_frame = state.get("prev_chunk_last_frame")
            if source_frame is None:
                raise RuntimeError("warp_as_history rollout is missing the previous chunk boundary frame.")
            source_frame = source_frame.to(device=device, dtype=torch.float32)

        if state.get("using_camera_warp", False):
            warp_video_chunk, visibility_chunk = self._render_camera_warp_chunk(
                state=state,
                chunk_index=chunk_index,
                source_frame=source_frame,
                device=device,
            )
        else:
            warp_video_tensor = state["warp_video_tensor"]
            visibility_mask = state["visibility_mask"]
            if warp_video_tensor is None or visibility_mask is None:
                raise RuntimeError("warp_as_history rollout is missing external warp state.")
            visibility_mask = visibility_mask.to(device=device, dtype=torch.float32)
            if frame_end > warp_video_tensor.shape[2]:
                raise RuntimeError(
                    "warp_as_history rollout is missing warp frames for chunk "
                    f"{chunk_index}: need frames [{frame_start}, {frame_end}), got {warp_video_tensor.shape[2]}."
                )
            visibility_chunk = visibility_mask[:, :, frame_start:frame_end].clone()

        if base_latents_history_short is None or base_latents_history_short.shape[2] < 1:
            raise RuntimeError("warp_as_history rollout is missing the official first-frame history prefix.")
        prefix_latent = base_latents_history_short[:, :, :1].detach().to(device=device, dtype=torch.float32)
        if bool(state.get("add_noise_to_warp_latents", True)) and not bool(
            state.get("image_history_prefix_noised", False)
        ):
            cached_prefix = state.get("warp_history_prefix_latent")
            if cached_prefix is None:
                noise_sigma_min = float(state.get("warp_noise_sigma_min", 0.111))
                noise_sigma_max = float(state.get("warp_noise_sigma_max", 0.135))
                rand_generator = generator[0] if isinstance(generator, list) else generator
                prefix_sigma = (
                    torch.rand(1, device=device, generator=rand_generator) * (noise_sigma_max - noise_sigma_min)
                    + noise_sigma_min
                ).to(dtype=prefix_latent.dtype)
                prefix_sigma = prefix_sigma.view(1, 1, 1, 1, 1)
                prefix_latent = (
                    prefix_sigma
                    * randn_tensor(prefix_latent.shape, generator=generator, device=device, dtype=prefix_latent.dtype)
                    + (1 - prefix_sigma) * prefix_latent
                )
                state["warp_history_prefix_latent"] = prefix_latent.detach()
            else:
                prefix_latent = cached_prefix.to(device=device, dtype=prefix_latent.dtype)

        if state.get("using_camera_warp", False):
            latents_mean, latents_std = self._latent_stats(device)
            _, warp_latents = self.prepare_video_latents(
                warp_video_chunk,
                latents_mean=latents_mean,
                latents_std=latents_std,
                num_latent_frames_per_chunk=WAH_NUM_LATENT_FRAMES_PER_CHUNK,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            warp_latents = self._add_noise_to_warp_history_latents(
                state=state,
                warp_latents=warp_latents,
                device=device,
                generator=generator,
            )
        else:
            warp_latents_tensor = self._prepare_external_warp_latents(
                state=state,
                device=device,
                generator=generator,
            )
            latent_start = chunk_index * WAH_NUM_LATENT_FRAMES_PER_CHUNK
            latent_end = latent_start + WAH_NUM_LATENT_FRAMES_PER_CHUNK
            if latent_end > warp_latents_tensor.shape[2]:
                raise RuntimeError(
                    "warp_as_history rollout is missing warp latent frames for chunk "
                    f"{chunk_index}: need latents [{latent_start}, {latent_end}), got {warp_latents_tensor.shape[2]}."
                )
            warp_latents = warp_latents_tensor[:, :, latent_start:latent_end].clone()

        prefix_latent = prefix_latent.to(dtype=warp_latents.dtype)
        visibility_latents = self._visibility_mask_to_history_latents(
            visibility_chunk,
            latent_frames=WAH_NUM_LATENT_FRAMES_PER_CHUNK,
            latent_height=int(state["height"] // self.vae_scale_factor_spatial),
            latent_width=int(state["width"] // self.vae_scale_factor_spatial),
            temporal_scale=int(self.vae_scale_factor_temporal),
        )

        prev_chunk_history_sizes = tuple(int(x) for x in state.get("prev_chunk_history_sizes", (0, 0, 0)))
        total_prev_history = sum(prev_chunk_history_sizes)
        long_size, mid_size, short_size = prev_chunk_history_sizes
        official_target_start = 1 + int(sum(WAH_HISTORY_SIZES))
        hidden_start = official_target_start
        if not state["rope_alignment"]:
            hidden_start += WAH_NUM_LATENT_FRAMES_PER_CHUNK
        indices_hidden_states = torch.arange(
            hidden_start,
            hidden_start + WAH_NUM_LATENT_FRAMES_PER_CHUNK,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)
        warp_indices = torch.arange(
            official_target_start,
            official_target_start + WAH_NUM_LATENT_FRAMES_PER_CHUNK,
            device=device,
            dtype=torch.long,
        )

        if total_prev_history > 0:
            prev_window = warp_latents.new_zeros(
                warp_latents.shape[0],
                warp_latents.shape[1],
                total_prev_history,
                warp_latents.shape[3],
                warp_latents.shape[4],
            )
            prev_visible_window = torch.zeros(
                warp_latents.shape[0],
                1,
                total_prev_history,
                warp_latents.shape[3],
                warp_latents.shape[4],
                device=device,
                dtype=torch.float32,
            )
            prev_history_latent_window = state.get("prev_history_latent_window")
            if chunk_index > 0:
                if prev_history_latent_window is None:
                    raise RuntimeError("warp_as_history rollout is missing previous latent history.")
                prev_history_latent_window = prev_history_latent_window.to(device=device, dtype=prev_window.dtype)
                available_history = min(int(prev_history_latent_window.shape[2]), int(total_prev_history))
                if available_history > 0:
                    prev_window[:, :, -available_history:] = prev_history_latent_window[:, :, -available_history:]
                    prev_visible_window[:, :, -available_history:] = 1.0
            elif short_size > 0 and base_latents_history_short.shape[2] > 1:
                fake_count = min(int(short_size), int(base_latents_history_short.shape[2] - 1))
                fake_short = base_latents_history_short[:, :, 1 : 1 + fake_count].detach()
                fake_short = fake_short.to(device=device, dtype=prev_window.dtype)
                short_end = int(long_size + mid_size + short_size)
                short_start = short_end - fake_count
                prev_window[:, :, short_start:short_end] = fake_short[:, :, -fake_count:]
                prev_visible_window[:, :, short_start:short_end] = 1.0
            prev_long, prev_mid, prev_short = prev_window.split(prev_chunk_history_sizes, dim=2)
            prev_visible_long, prev_visible_mid, prev_visible_short = prev_visible_window.split(
                prev_chunk_history_sizes,
                dim=2,
            )
            history_start = official_target_start - total_prev_history
            prev_indices = torch.arange(
                history_start,
                history_start + total_prev_history,
                device=device,
                dtype=torch.long,
            )
            prev_long_indices, prev_mid_indices, prev_short_indices = prev_indices.split(prev_chunk_history_sizes, dim=0)
        else:
            prev_long = prev_mid = prev_short = None
            prev_visible_long = prev_visible_mid = prev_visible_short = None
            prev_long_indices = prev_mid_indices = prev_short_indices = None

        if prefix_latent.shape[0] != warp_latents.shape[0] or prefix_latent.shape[-2:] != warp_latents.shape[-2:]:
            raise RuntimeError(
                "warp_as_history first-frame prefix shape does not match warp history latents: "
                f"prefix={tuple(prefix_latent.shape)}, warp={tuple(warp_latents.shape)}."
            )

        short_indices_parts = [torch.zeros(1, device=device, dtype=torch.long)]
        short_history_parts = [prefix_latent]
        short_visible_parts = [] if state["visible_token_drop"] else None
        if short_visible_parts is not None:
            short_visible_parts.append(
                torch.ones(
                    warp_latents.shape[0],
                    1,
                    1,
                    warp_latents.shape[3],
                    warp_latents.shape[4],
                    device=device,
                    dtype=torch.float32,
                )
            )
        if short_size > 0:
            short_indices_parts.append(prev_short_indices)
            short_history_parts.append(prev_short)
            if short_visible_parts is not None:
                short_visible_parts.append(prev_visible_short)
        short_indices_parts.append(warp_indices)
        short_history_parts.append(warp_latents)
        if short_visible_parts is not None:
            short_visible_parts.append(visibility_latents)

        return {
            "indices_hidden_states": indices_hidden_states,
            "indices_latents_history_short": torch.cat(short_indices_parts, dim=0).unsqueeze(0),
            "indices_latents_history_mid": prev_mid_indices.unsqueeze(0) if mid_size > 0 else None,
            "indices_latents_history_long": prev_long_indices.unsqueeze(0) if long_size > 0 else None,
            "latents_history_short": torch.cat(short_history_parts, dim=2),
            "latents_history_mid": prev_mid if mid_size > 0 else None,
            "latents_history_long": prev_long if long_size > 0 else None,
            "history_visible_mask_short": None
            if short_visible_parts is None
            else torch.cat(short_visible_parts, dim=2),
            "history_visible_mask_mid": None
            if not state["visible_token_drop"] or mid_size <= 0
            else prev_visible_mid,
            "history_visible_mask_long": None
            if not state["visible_token_drop"] or long_size <= 0
            else prev_visible_long,
        }

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str] | None,
        image: PipelineImageInput,
        warp_video: Any | None = None,
        *,
        warp_visibility_mask: Any | None = None,
        camera_poses: torch.Tensor | np.ndarray | None = None,
        target_intrinsics: torch.Tensor | np.ndarray | None = None,
        lora_path: str | Path | None = None,
        visible_token_drop: bool = True,
        rope_alignment: bool = True,
        warp_invisible_fill: str = "mean_first_frame",
        height: int = 384,
        width: int = 640,
        num_frames: int = WAH_NUM_FRAMES,
        negative_prompt: str | list[str] | None = WAH_NEGATIVE_PROMPT,
        generator: torch.Generator | list[torch.Generator] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        output_type: str | None = "np",
        return_dict: bool = True,
        num_videos_per_prompt: int = 1,
        add_noise_to_image_latents: bool = False,
        add_noise_to_warp_latents: bool = True,
        warp_noise_sigma_min: float = 0.111,
        warp_noise_sigma_max: float = 0.135,
        is_amplify_first_chunk: bool = True,
        lora_prompt_trigger: str | None = None,
        prev_chunk_history_sizes: list[int] | tuple[int, int, int] = WAH_PREV_CHUNK_HISTORY_SIZES,
        camera_control_translation_scale: float = CAMERA_CONTROL_DEFAULT_TRANSLATION_SCALE,
        camera_control_translation_scale_use_first_frame_depth: bool = (
            CAMERA_CONTROL_DEFAULT_TRANSLATION_SCALE_USE_FIRST_FRAME_DEPTH
        ),
        camera_control_warp_invisible_fill: str = CAMERA_CONTROL_DEFAULT_WARP_INVISIBLE_FILL,
        camera_control_warp_render_mode: str = CAMERA_CONTROL_DEFAULT_WARP_RENDER_MODE,
        camera_control_warp_target_fill_radius: int = CAMERA_CONTROL_DEFAULT_WARP_TARGET_FILL_RADIUS,
        camera_control_warp_target_fill_min_neighbors: int = CAMERA_CONTROL_DEFAULT_WARP_TARGET_FILL_MIN_NEIGHBORS,
        camera_control_mesh_break_mode: str = CAMERA_CONTROL_DEFAULT_MESH_BREAK_MODE,
        camera_control_mesh_depth_rtol: float = CAMERA_CONTROL_DEFAULT_MESH_DEPTH_RTOL,
        camera_control_mesh_normal_tol_deg: float = CAMERA_CONTROL_DEFAULT_MESH_NORMAL_TOL_DEG,
        camera_control_pi3x_keyframe_memory: bool = CAMERA_CONTROL_DEFAULT_PI3X_KEYFRAME_MEMORY,
    ) -> Any:
        using_camera_warp = warp_video is None
        camera_renderer = None
        camera_first_chunk_rendered = None
        if using_camera_warp:
            if camera_poses is None:
                raise ValueError("Either warp_video or camera_poses must be provided.")
            if warp_visibility_mask is not None:
                raise ValueError("warp_visibility_mask is only supported when warp_video is provided.")

            window_num_frames = (WAH_NUM_LATENT_FRAMES_PER_CHUNK - 1) * self.vae_scale_factor_temporal + 1
            device = self._wah_execution_device()
            image = center_crop_resize_first_frame(image, height=int(height), width=int(width))
            source_image_tensor = self.video_processor.preprocess(
                image,
                height=int(height),
                width=int(width),
            ).to(device=device, dtype=torch.float32)
            renderer = self._get_camera_warp_renderer(
                camera_control_warp_render_mode=camera_control_warp_render_mode,
                camera_control_warp_target_fill_radius=camera_control_warp_target_fill_radius,
                camera_control_warp_target_fill_min_neighbors=camera_control_warp_target_fill_min_neighbors,
                camera_control_mesh_break_mode=camera_control_mesh_break_mode,
                camera_control_mesh_depth_rtol=float(camera_control_mesh_depth_rtol),
                camera_control_mesh_normal_tol_deg=float(camera_control_mesh_normal_tol_deg),
            )
            first_chunk_intrinsics = self._slice_frame_sequence(
                target_intrinsics,
                0,
                window_num_frames,
                "target_intrinsics",
            )
            camera_first_chunk_rendered = renderer.render(
                image_tensor=source_image_tensor,
                camera_poses=camera_poses,
                height=int(height),
                width=int(width),
                num_frames=window_num_frames,
                device=device,
                target_intrinsics=first_chunk_intrinsics,
                translation_scale=float(camera_control_translation_scale),
                translation_scale_use_first_frame_depth=bool(
                    camera_control_translation_scale_use_first_frame_depth
                ),
                invisible_fill_mode=str(camera_control_warp_invisible_fill),
                render_mode=str(camera_control_warp_render_mode),
                target_fill_radius=int(camera_control_warp_target_fill_radius),
                target_fill_min_neighbors=int(camera_control_warp_target_fill_min_neighbors),
                mesh_break_mode=str(camera_control_mesh_break_mode),
            )
            self._last_camera_warp = camera_first_chunk_rendered
            camera_renderer = renderer
            warp_invisible_fill = str(camera_control_warp_invisible_fill)
        elif camera_poses is not None:
            raise ValueError("Pass either warp_video or camera_poses, not both.")

        if lora_prompt_trigger is None:
            lora_prompt_trigger = CAMERA_CONTROL_PROMPT_TRIGGER if using_camera_warp else WAH_PROMPT_TRIGGER

        self._check_minimal_inputs(
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_videos_per_prompt=num_videos_per_prompt,
        )

        normalized_lora_path = _normalize_optional_lora_path(lora_path)
        lora_active = self._configure_wah_lora(normalized_lora_path)
        prompt_for_pipe = (
            self._add_prompt_trigger(prompt, lora_prompt_trigger) if lora_active and prompt_embeds is None else prompt
        )
        attention_kwargs = (
            {"history_visible_token_threshold": WAH_VISIBLE_TOKEN_THRESHOLD} if bool(visible_token_drop) else None
        )

        self._wah_state = self._prepare_warp_state(
            image=image,
            warp_video=warp_video,
            warp_visibility_mask=warp_visibility_mask,
            height=int(height),
            width=int(width),
            num_frames=int(num_frames),
            warp_invisible_fill=str(warp_invisible_fill),
            visible_token_drop=bool(visible_token_drop),
            rope_alignment=bool(rope_alignment),
            prev_chunk_history_sizes=prev_chunk_history_sizes,
            add_noise_to_warp_latents=bool(add_noise_to_warp_latents),
            warp_noise_sigma_min=float(warp_noise_sigma_min),
            warp_noise_sigma_max=float(warp_noise_sigma_max),
            image_history_prefix_noised=bool(add_noise_to_image_latents),
            generator=generator,
            lora_active=bool(lora_active),
        )
        if using_camera_warp:
            self._wah_state.update(
                {
                    "using_camera_warp": True,
                    "camera_renderer": camera_renderer,
                    "camera_poses": camera_poses,
                    "target_intrinsics": target_intrinsics,
                    "camera_first_chunk_rendered": camera_first_chunk_rendered,
                    "camera_first_frame_geometry": camera_first_chunk_rendered["geometry"],
                    "camera_translation_effective_scale": float(
                        camera_first_chunk_rendered.get(
                            "camera_translation_effective_scale",
                            camera_control_translation_scale,
                        )
                    ),
                    "camera_control_translation_scale": float(camera_control_translation_scale),
                    "camera_control_translation_scale_use_first_frame_depth": bool(
                        camera_control_translation_scale_use_first_frame_depth
                    ),
                    "camera_control_warp_invisible_fill": str(camera_control_warp_invisible_fill),
                    "camera_control_warp_render_mode": str(camera_control_warp_render_mode),
                    "camera_control_warp_target_fill_radius": int(camera_control_warp_target_fill_radius),
                    "camera_control_warp_target_fill_min_neighbors": int(
                        camera_control_warp_target_fill_min_neighbors
                    ),
                    "camera_control_mesh_break_mode": str(camera_control_mesh_break_mode),
                    "pi3x_keyframe_images": [source_image_tensor.detach().float().cpu()]
                    if bool(camera_control_pi3x_keyframe_memory)
                    else None,
                    "pi3x_keyframe_last_decoded_chunk": -1,
                    "pi3x_keyframe_memory_enabled": bool(camera_control_pi3x_keyframe_memory),
                    "pi3x_keyframe_counts": [],
                    "pi3x_keyframe_intrinsic_smoothing_stats": [],
                    "pi3x_keyframe_scale_alignment_stats": [],
                    "camera_warp_chunks": {0: camera_first_chunk_rendered},
                }
            )
        original_vae_decode = self.vae.decode

        def wah_wrapped_decode(*args, **kwargs):
            decoded = original_vae_decode(*args, **kwargs)
            state = getattr(self, "_wah_state", None)
            decoded_video = decoded[0] if isinstance(decoded, tuple) else decoded.sample
            if (
                state is not None
                and isinstance(decoded_video, torch.Tensor)
                and decoded_video.ndim == 5
                and decoded_video.shape[2] >= 1
            ):
                boundary_frame = _display_boundary_frame(decoded_video[:, :, -1])
                state["prev_chunk_last_frame"] = boundary_frame
                pi3x_keyframe_images = state.get("pi3x_keyframe_images")
                if pi3x_keyframe_images is not None:
                    decoded_chunk_index = int(state.get("chunk_index", 0)) - 1
                    last_decoded_chunk = int(state.get("pi3x_keyframe_last_decoded_chunk", -1))
                    if decoded_chunk_index >= 0 and decoded_chunk_index > last_decoded_chunk:
                        pi3x_keyframe_images.append(boundary_frame)
                        state["pi3x_keyframe_last_decoded_chunk"] = decoded_chunk_index
            return decoded

        self.vae.decode = wah_wrapped_decode
        try:
            return super().__call__(
                prompt=prompt_for_pipe,
                negative_prompt=negative_prompt,
                height=int(height),
                width=int(width),
                num_frames=int(num_frames),
                guidance_scale=1.0,
                num_videos_per_prompt=num_videos_per_prompt,
                generator=generator,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                output_type=output_type,
                return_dict=return_dict,
                attention_kwargs=attention_kwargs,
                image=image,
                add_noise_to_image_latents=bool(add_noise_to_image_latents),
                image_noise_sigma_min=0.111,
                image_noise_sigma_max=0.135,
                history_sizes=list(WAH_HISTORY_SIZES),
                num_latent_frames_per_chunk=WAH_NUM_LATENT_FRAMES_PER_CHUNK,
                keep_first_frame=True,
                is_skip_first_chunk=False,
                is_enable_stage2=True,
                pyramid_num_stages=WAH_PYRAMID_NUM_STAGES,
                pyramid_num_inference_steps_list=list(WAH_PYRAMID_STEPS),
                use_zero_init=False,
                zero_steps=0,
                is_amplify_first_chunk=bool(is_amplify_first_chunk),
            )
        finally:
            self.vae.decode = original_vae_decode
            self._wah_state = None
            self._set_wah_lora_enabled(bool(lora_active))

    def stage2_sample(
        self,
        latents: torch.Tensor = None,
        pyramid_num_stages: int = None,
        pyramid_num_inference_steps_list: list[int] = None,
        prompt_embeds: torch.Tensor = None,
        negative_prompt_embeds: torch.Tensor = None,
        guidance_scale: float | None = 5.0,
        indices_hidden_states: torch.Tensor = None,
        indices_latents_history_short: torch.Tensor = None,
        indices_latents_history_mid: torch.Tensor = None,
        indices_latents_history_long: torch.Tensor = None,
        latents_history_short: torch.Tensor = None,
        latents_history_mid: torch.Tensor = None,
        latents_history_long: torch.Tensor = None,
        attention_kwargs: dict | None = None,
        device: torch.device | None = None,
        transformer_dtype: torch.dtype = None,
        generator: torch.Generator | None = None,
        use_zero_init: bool | None = True,
        zero_steps: int | None = 1,
        is_amplify_first_chunk: bool = False,
        callback_on_step_end: Callable[[int, int], None] | PipelineCallback | MultiPipelineCallbacks | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        progress_bar=None,
    ):
        state = getattr(self, "_wah_state", None)
        if state is None:
            return super().stage2_sample(
                latents=latents,
                pyramid_num_stages=pyramid_num_stages,
                pyramid_num_inference_steps_list=pyramid_num_inference_steps_list,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=guidance_scale,
                indices_hidden_states=indices_hidden_states,
                indices_latents_history_short=indices_latents_history_short,
                indices_latents_history_mid=indices_latents_history_mid,
                indices_latents_history_long=indices_latents_history_long,
                latents_history_short=latents_history_short,
                latents_history_mid=latents_history_mid,
                latents_history_long=latents_history_long,
                attention_kwargs=attention_kwargs,
                device=device,
                transformer_dtype=transformer_dtype,
                generator=generator,
                use_zero_init=use_zero_init,
                zero_steps=zero_steps,
                is_amplify_first_chunk=is_amplify_first_chunk,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                progress_bar=progress_bar,
            )

        batch_size, num_channel, num_frames, height, width = latents.shape
        latents = latents.permute(0, 2, 1, 3, 4).reshape(batch_size * num_frames, num_channel, height, width)
        for _ in range(pyramid_num_stages - 1):
            height //= 2
            width //= 2
            latents = F.interpolate(latents, size=(height, width), mode="bilinear") * 2
        latents = latents.reshape(batch_size, num_frames, num_channel, height, width).permute(0, 2, 1, 3, 4)

        batch_size = latents.shape[0]
        start_point_list = None
        if self.config.is_distilled:
            start_point_list = [latents]

        i = 0
        try:
            for i_s in range(pyramid_num_stages):
                self._set_wah_lora_enabled(bool(state["lora_active"]) and i_s == 0)

                patch_size = self.transformer.config.patch_size
                image_seq_len = (latents.shape[-1] * latents.shape[-2] * latents.shape[-3]) // (
                    patch_size[0] * patch_size[1] * patch_size[2]
                )
                mu = calculate_shift(
                    image_seq_len,
                    self.scheduler.config.get("base_image_seq_len", 256),
                    self.scheduler.config.get("max_image_seq_len", 4096),
                    self.scheduler.config.get("base_shift", 0.5),
                    self.scheduler.config.get("max_shift", 1.15),
                )
                self.scheduler.set_timesteps(
                    pyramid_num_inference_steps_list[i_s],
                    i_s,
                    device=device,
                    mu=mu,
                    is_amplify_first_chunk=is_amplify_first_chunk,
                )
                timesteps = self.scheduler.timesteps

                if i_s > 0:
                    height *= 2
                    width *= 2
                    num_frames = latents.shape[2]
                    latents = latents.permute(0, 2, 1, 3, 4).reshape(
                        batch_size * num_frames, num_channel, height // 2, width // 2
                    )
                    latents = F.interpolate(latents, size=(height, width), mode="nearest")
                    latents = latents.reshape(batch_size, num_frames, num_channel, height, width).permute(
                        0, 2, 1, 3, 4
                    )

                    ori_sigma = 1 - self.scheduler.ori_start_sigmas[i_s]
                    gamma = self.scheduler.config.gamma
                    alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)
                    beta = alpha * (1 - ori_sigma) / math.sqrt(gamma)

                    batch_size, channel, num_frames, height, width = latents.shape
                    noise = self.sample_block_noise(
                        batch_size,
                        channel,
                        num_frames,
                        height,
                        width,
                        patch_size,
                        device,
                        generator,
                    )
                    noise = noise.to(device=device, dtype=transformer_dtype)
                    latents = alpha * latents + beta * noise

                    if self.config.is_distilled:
                        start_point_list.append(latents)

                if i_s == 0:
                    pyramid_base_histories = self._build_pyramid_base_histories(
                        state=state,
                        device=device,
                        history_dtype=latents_history_short.dtype,
                        generator=generator,
                        base_latents_history_short=latents_history_short,
                    )

                for idx, t in enumerate(timesteps):
                    if i_s == 0:
                        current_indices_hidden_states = pyramid_base_histories["indices_hidden_states"]
                        current_indices_latents_history_short = pyramid_base_histories[
                            "indices_latents_history_short"
                        ]
                        current_indices_latents_history_mid = pyramid_base_histories["indices_latents_history_mid"]
                        current_indices_latents_history_long = pyramid_base_histories["indices_latents_history_long"]
                        current_latents_history_short = pyramid_base_histories["latents_history_short"]
                        current_latents_history_mid = pyramid_base_histories["latents_history_mid"]
                        current_latents_history_long = pyramid_base_histories["latents_history_long"]
                        current_history_visible_mask_short = pyramid_base_histories["history_visible_mask_short"]
                        current_history_visible_mask_mid = pyramid_base_histories["history_visible_mask_mid"]
                        current_history_visible_mask_long = pyramid_base_histories["history_visible_mask_long"]
                    else:
                        current_indices_hidden_states = indices_hidden_states
                        current_indices_latents_history_short = indices_latents_history_short
                        current_indices_latents_history_mid = indices_latents_history_mid
                        current_indices_latents_history_long = indices_latents_history_long
                        current_latents_history_short = latents_history_short
                        current_latents_history_mid = latents_history_mid
                        current_latents_history_long = latents_history_long
                        current_history_visible_mask_short = None
                        current_history_visible_mask_mid = None
                        current_history_visible_mask_long = None

                    timestep = t.expand(latents.shape[0]).to(torch.int64)
                    model_latents = latents.to(transformer_dtype)

                    with self.transformer.cache_context("cond"):
                        noise_pred = self.transformer(
                            hidden_states=model_latents,
                            timestep=timestep,
                            encoder_hidden_states=prompt_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                            indices_hidden_states=current_indices_hidden_states,
                            indices_latents_history_short=current_indices_latents_history_short,
                            indices_latents_history_mid=current_indices_latents_history_mid,
                            indices_latents_history_long=current_indices_latents_history_long,
                            latents_history_short=_optional_to_dtype(
                                current_latents_history_short, transformer_dtype
                            ),
                            latents_history_mid=_optional_to_dtype(current_latents_history_mid, transformer_dtype),
                            latents_history_long=_optional_to_dtype(current_latents_history_long, transformer_dtype),
                            history_visible_mask_short=current_history_visible_mask_short,
                            history_visible_mask_mid=current_history_visible_mask_mid,
                            history_visible_mask_long=current_history_visible_mask_long,
                        )[0]

                    if self.do_classifier_free_guidance:
                        with self.transformer.cache_context("uncond"):
                            noise_uncond = self.transformer(
                                hidden_states=model_latents,
                                timestep=timestep,
                                encoder_hidden_states=negative_prompt_embeds,
                                attention_kwargs=attention_kwargs,
                                return_dict=False,
                                indices_hidden_states=current_indices_hidden_states,
                                indices_latents_history_short=current_indices_latents_history_short,
                                indices_latents_history_mid=current_indices_latents_history_mid,
                                indices_latents_history_long=current_indices_latents_history_long,
                                latents_history_short=_optional_to_dtype(
                                    current_latents_history_short, transformer_dtype
                                ),
                                latents_history_mid=_optional_to_dtype(current_latents_history_mid, transformer_dtype),
                                latents_history_long=_optional_to_dtype(
                                    current_latents_history_long, transformer_dtype
                                ),
                                history_visible_mask_short=current_history_visible_mask_short,
                                history_visible_mask_mid=current_history_visible_mask_mid,
                                history_visible_mask_long=current_history_visible_mask_long,
                            )[0]

                        if self.config.is_cfg_zero_star:
                            noise_pred_text = noise_pred
                            positive_flat = noise_pred_text.view(batch_size, -1)
                            negative_flat = noise_uncond.view(batch_size, -1)

                            alpha = optimized_scale(positive_flat, negative_flat)
                            alpha = alpha.view(batch_size, *([1] * (len(noise_pred_text.shape) - 1)))
                            alpha = alpha.to(noise_pred_text.dtype)

                            if (i_s == 0 and idx <= zero_steps) and use_zero_init:
                                noise_pred = noise_pred_text * 0.0
                            else:
                                noise_pred = (
                                    noise_uncond * alpha + guidance_scale * (noise_pred_text - noise_uncond * alpha)
                                )
                        else:
                            noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

                    latents = self.scheduler.step(
                        noise_pred,
                        t,
                        latents,
                        generator=generator,
                        return_dict=False,
                        cur_sampling_step=idx,
                        dmd_noisy_tensor=start_point_list[i_s] if start_point_list is not None else None,
                        dmd_sigmas=self.scheduler.sigmas,
                        dmd_timesteps=self.scheduler.timesteps,
                        all_timesteps=timesteps,
                    )[0]

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                    if progress_bar is not None:
                        progress_bar.update()

                    if XLA_AVAILABLE:
                        xm.mark_step()

                    i += 1
        finally:
            self._set_wah_lora_enabled(bool(state["lora_active"]))

        total_prev_history = sum(int(x) for x in state.get("prev_chunk_history_sizes", (0, 0, 0)))
        if total_prev_history > 0:
            prev_history = state.get("prev_history_latent_window")
            current_history = latents.detach()
            if prev_history is not None:
                prev_history = prev_history.to(device=current_history.device, dtype=current_history.dtype)
                current_history = torch.cat([prev_history, current_history], dim=2)
            state["prev_history_latent_window"] = current_history[:, :, -total_prev_history:].detach()
        state["chunk_index"] = int(state.get("chunk_index", 0)) + 1
        return latents
