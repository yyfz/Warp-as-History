#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import importlib
import importlib.util
import io
import json
import math
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TAEHV_ROOT = REPO_ROOT / "third_party" / "taehv"
if TAEHV_ROOT.is_dir() and str(TAEHV_ROOT) not in sys.path:
    sys.path.insert(0, str(TAEHV_ROOT))

DEFAULT_MODEL = "checkpoints/helios-distilled"
DEFAULT_WAH_LORA = "checkpoints/warp-as-history/visible_lora_state_step1000.safetensors"
DEFAULT_EFFICIENT_WAH_LORA = (
    "checkpoints/warp-as-history/visible_lora_state_step1000_efficient_patchmid.pt"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "web_realtime_demo"
DEFAULT_EFFICIENT_OUTPUT_DIR = REPO_ROOT / "runs" / "web_realtime_demo_efficient_realtime"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal realtime Warp-as-History web demo.")
    parser.add_argument(
        "--preset",
        choices=["normal", "efficient_realtime"],
        default="normal",
        help="normal uses the standard WAH recipe; efficient_realtime matches the realtime H200 preset.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--model_path", default=DEFAULT_MODEL, help=argparse.SUPPRESS)
    parser.add_argument("--lora_path", default=DEFAULT_WAH_LORA, help=argparse.SUPPRESS)
    parser.add_argument("--no_lora", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--height", type=int, default=384, help=argparse.SUPPRESS)
    parser.add_argument("--width", type=int, default=640, help=argparse.SUPPRESS)
    parser.add_argument("--fps", type=int, default=16, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=42, help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cuda", help=argparse.SUPPRESS)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto", help=argparse.SUPPRESS)
    parser.add_argument("--output_dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--preload", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--enable_compile", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--matmul_precision", choices=["highest", "high", "medium"], default="", help=argparse.SUPPRESS)
    parser.add_argument("--disable_progress", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--pyramid_steps", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no_amplify_first_chunk", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--warp_history_downsample_mode",
        choices=["short", "patch_mid"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--warp_history_spatial_downsample", type=int, default=2, help=argparse.SUPPRESS)
    parser.add_argument("--no_visible_token_drop", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--visible_token_mode", choices=["mask", "drop"], default="mask", help=argparse.SUPPRESS)
    parser.add_argument("--visible_token_threshold", type=float, default=0.1, help=argparse.SUPPRESS)
    parser.add_argument("--visible_mask_erosion_radius", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--visible_latent_fill", choices=["none", "zero", "noise", "prefix"], default="none", help=argparse.SUPPRESS)
    parser.add_argument("--no_pi3x_keyframe_memory", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--pi3x_keyframe_stride", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--camera_warp_render_mode", choices=["target_fill", "splat"], default="splat", help=argparse.SUPPRESS)
    parser.add_argument("--camera_pi3_pixel_limit", type=int, default=255000, help=argparse.SUPPRESS)
    parser.add_argument("--camera_mesh_samples_per_axis", type=int, default=4, help=argparse.SUPPRESS)
    parser.add_argument("--taehv_vae_mode", choices=["off", "decode", "full"], default="off", help=argparse.SUPPRESS)
    parser.add_argument("--taehv_checkpoint", default=str(REPO_ROOT / "checkpoints" / "taehv" / "taew2_1.pth"), help=argparse.SUPPRESS)
    parser.add_argument("--enable_official_kernels", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--enable_optional_attention", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def parse_pyramid_steps(text: str) -> list[int]:
    values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("--pyramid_steps must contain at least one integer.")
    if any(value <= 0 for value in values):
        raise ValueError("--pyramid_steps values must be positive.")
    return values


def json_response(handler: BaseHTTPRequestHandler, payload: dict[str, Any], status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def text_response(
    handler: BaseHTTPRequestHandler,
    body: str,
    *,
    status: int = 200,
    content_type: str = "text/html; charset=utf-8",
) -> None:
    data = body.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def patch_diffusers_kernel_attr_resolution() -> None:
    try:
        import diffusers.models.attention_dispatch as attention_dispatch
    except Exception:
        return

    original_resolve = getattr(attention_dispatch, "_resolve_kernel_attr", None)
    if original_resolve is None or getattr(original_resolve, "_wah_kernel_attr_patch", False):
        return

    def resolve_kernel_attr(module: Any, attr_path: str) -> Any:
        try:
            return original_resolve(module, attr_path)
        except AttributeError:
            parts = attr_path.split(".")
            if len(parts) < 2:
                raise
            submodule = importlib.import_module(f"{module.__name__}.{parts[0]}")
            setattr(module, parts[0], submodule)
            target = submodule
            for attr in parts[1:]:
                target = getattr(target, attr)
            return target

    resolve_kernel_attr._wah_kernel_attr_patch = True
    attention_dispatch._resolve_kernel_attr = resolve_kernel_attr


def preload_cached_flash_attn3_hub_kernel() -> None:
    try:
        import diffusers.models.attention_dispatch as attention_dispatch
    except Exception:
        return

    registry = getattr(attention_dispatch, "_HUB_KERNELS_REGISTRY", None)
    backend_names = getattr(attention_dispatch, "AttentionBackendName", None)
    if registry is None or backend_names is None:
        return
    backend = getattr(backend_names, "_FLASH_3_HUB", None)
    if backend is None or backend not in registry:
        return
    config = registry[backend]
    if getattr(config, "kernel_fn", None) is not None:
        return

    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / "models--kernels-community--flash-attn3"
    candidates = sorted(cache_root.glob("snapshots/*/build/torch*-cu*-x86_64-linux/__init__.py"), reverse=True)
    for init_file in candidates:
        package_dir = init_file.parent
        package_name = f"_wah_cached_flash_attn3_{abs(hash(str(package_dir)))}"
        try:
            spec = importlib.util.spec_from_file_location(
                package_name,
                str(init_file),
                submodule_search_locations=[str(package_dir)],
            )
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[package_name] = module
            spec.loader.exec_module(module)
            interface = importlib.import_module(f"{package_name}.flash_attn_interface")
            config.kernel_fn = module.flash_attn_func
            config.wrapped_forward_fn = interface._flash_attn_forward
            config.wrapped_backward_fn = interface._flash_attn_backward
            print(
                json.dumps({"event": "preload_cached_flash_attn3_hub_kernel", "path": str(package_dir)}),
                flush=True,
            )
            return
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "event": "preload_cached_flash_attn3_hub_kernel_error",
                        "path": str(package_dir),
                        "error": repr(exc),
                    }
                ),
                flush=True,
            )


def set_transformer_attention_backend(transformer: Any, backend: str) -> None:
    if backend == "wah_local_fa3":
        count = 0
        for module in transformer.modules():
            processor = getattr(module, "processor", None)
            if processor is not None and hasattr(processor, "_attention_backend"):
                processor._attention_backend = backend
                count += 1
        if count <= 0:
            raise RuntimeError("No Helios attention processors were found for wah_local_fa3")
        return
    transformer.set_attention_backend(backend)


_NUMPY: Any | None = None


def numpy_module() -> Any:
    global _NUMPY
    if _NUMPY is None:
        import numpy as np

        _NUMPY = np
    return _NUMPY


def _rotation_x(angle: float) -> np.ndarray:
    np = numpy_module()
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


def _rotation_y(angle: float) -> np.ndarray:
    np = numpy_module()
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def _rotation_z(angle: float) -> np.ndarray:
    np = numpy_module()
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _camera_delta(
    *,
    translation: np.ndarray,
    yaw: float,
    pitch: float,
    roll: float,
) -> np.ndarray:
    np = numpy_module()
    delta = np.eye(4, dtype=np.float32)
    delta[:3, :3] = _rotation_z(roll) @ _rotation_y(yaw) @ _rotation_x(pitch)
    delta[:3, 3] = translation.astype(np.float32, copy=False)
    return delta


def _control_delta(active: set[str], rotation_degrees: float) -> tuple[np.ndarray, float, float, float]:
    np = numpy_module()
    translation = np.zeros(3, dtype=np.float32)
    if "strafe_left" in active:
        translation[0] -= 1.0
    if "strafe_right" in active:
        translation[0] += 1.0
    if "rise" in active:
        translation[1] += 1.0
    if "descend" in active:
        translation[1] -= 1.0
    if "forward" in active:
        translation[2] += 1.0
    if "backward" in active:
        translation[2] -= 1.0
    norm = float(np.linalg.norm(translation))
    if norm > 1.0e-6:
        translation /= norm

    angle = math.radians(float(rotation_degrees))
    yaw = 0.0
    pitch = 0.0
    roll = 0.0
    if "yaw_left" in active:
        yaw -= angle
    if "yaw_right" in active:
        yaw += angle
    if "pitch_up" in active:
        pitch += angle
    if "pitch_down" in active:
        pitch -= angle
    if "roll_left" in active:
        roll -= angle
    if "roll_right" in active:
        roll += angle
    return translation, yaw, pitch, roll


def build_camera_chunk(
    start_pose: np.ndarray,
    *,
    active: set[str],
    rotation_degrees: float,
    window_num_frames: int,
    include_start: bool,
) -> tuple[np.ndarray, np.ndarray]:
    np = numpy_module()
    translation, yaw, pitch, roll = _control_delta(active, rotation_degrees)
    if include_start:
        alphas = np.linspace(0.0, 1.0, int(window_num_frames), dtype=np.float32)
    else:
        alphas = np.linspace(1.0 / int(window_num_frames), 1.0, int(window_num_frames), dtype=np.float32)
    poses = []
    for alpha in alphas:
        delta = _camera_delta(
            translation=translation * float(alpha),
            yaw=yaw * float(alpha),
            pitch=pitch * float(alpha),
            roll=roll * float(alpha),
        )
        poses.append((start_pose @ delta).astype(np.float32))
    end_pose = (start_pose @ _camera_delta(translation=translation, yaw=yaw, pitch=pitch, roll=roll)).astype(np.float32)
    return np.stack(poses, axis=0), end_pose


def write_mse_segment(path: Path, frames: list[Any], fps: int) -> None:
    import imageio.v2 as imageio
    from scripts.infer_warp_as_history import frame_to_uint8

    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(path),
        fps=int(fps),
        codec="libx264",
        macro_block_size=1,
        output_params=[
            "-movflags",
            "frag_keyframe+empty_moov+default_base_moof",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-profile:v",
            "baseline",
            "-level",
            "3.0",
            "-pix_fmt",
            "yuv420p",
        ],
    ) as writer:
        for frame in frames:
            writer.append_data(frame_to_uint8(frame))


@dataclass
class AppConfig:
    model_path: str
    lora_path: str | None
    height: int
    width: int
    fps: int
    seed: int
    device: str
    dtype: str
    output_dir: Path
    enable_compile: bool
    matmul_precision: str
    disable_progress: bool
    pyramid_num_inference_steps_list: list[int] | None
    amplify_first_chunk: bool
    warp_history_downsample_mode: str
    visible_token_drop: bool
    visible_token_threshold: float
    pi3x_keyframe_memory: bool
    camera_warp_render_mode: str
    camera_pi3_pixel_limit: int
    camera_mesh_samples_per_axis: int
    taehv_vae_mode: str
    taehv_checkpoint: Path | None
    enable_official_kernels: bool


@dataclass
class SessionState:
    pipe: Any = None
    wah_state: dict[str, Any] | None = None
    prompt: str = ""
    current_pose: np.ndarray | None = None
    session_id: str = ""
    generated_chunks: int = 0
    last_video_path: Path | None = None
    history_items: list[dict[str, Any]] | None = None
    window_num_frames: int = 33


@dataclass
class ContinuousState:
    running: bool = False
    worker: threading.Thread | None = None
    stop_event: threading.Event | None = None
    active: set[str] | None = None
    translation_scale: float = 0.1
    rotation_degrees: float = 4.0
    target_buffer: int = 2
    playing_index: int = -1
    generating_chunk: int | None = None
    last_latency: float | None = None
    last_profile: dict[str, Any] | None = None
    error: str = ""


class WarpControlApp:
    def __init__(self, config: AppConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.state = SessionState()
        self.lock = threading.Lock()
        self.control_lock = threading.Lock()
        self.continuous = ContinuousState(active=set())

    def load_pipeline(self) -> Any:
        if self.state.pipe is not None:
            return self.state.pipe
        import torch
        from warp_as_history import WarpAsHistoryPipeline
        from scripts.infer_warp_as_history import torch_dtype_from_arg

        if self.config.matmul_precision:
            torch.set_float32_matmul_precision(self.config.matmul_precision)
        dtype = torch_dtype_from_arg(self.config.dtype, self.config.device)
        pipe = WarpAsHistoryPipeline.from_pretrained(self.config.model_path, torch_dtype=dtype).to(self.config.device)
        if self.config.enable_official_kernels:
            if not self.config.enable_compile:
                from helios.modules.helios_kernels import replace_all_norms_with_flash_norms, replace_rmsnorm_with_fp32

                pipe.transformer = replace_rmsnorm_with_fp32(pipe.transformer)
                pipe.transformer = replace_all_norms_with_flash_norms(pipe.transformer)
            if hasattr(pipe.transformer, "set_attention_backend"):
                candidate_backends = (
                    ["_flash_3_hub", "wah_local_fa3", "_native_flash"]
                    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9
                    else ["flash_hub", "wah_local_fa3", "_native_flash"]
                )
                backend = None
                last_error = None
                for candidate_backend in candidate_backends:
                    try:
                        set_transformer_attention_backend(pipe.transformer, candidate_backend)
                        backend = candidate_backend
                        break
                    except Exception as exc:
                        last_error = exc
                        print(
                            json.dumps(
                                {
                                    "event": "official_kernels_backend_error",
                                    "attention_backend": candidate_backend,
                                    "error": repr(exc),
                                }
                            ),
                            flush=True,
                        )
                if backend is None:
                    raise RuntimeError(f"Could not set any official attention backend: {candidate_backends}") from last_error
                print(json.dumps({"event": "official_kernels", "attention_backend": backend}), flush=True)
        if self.config.disable_progress:
            pipe.set_progress_bar_config(disable=True)
        if self.config.enable_compile:
            torch.backends.cudnn.benchmark = True
            pipe.text_encoder.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
            if self.config.taehv_vae_mode != "full":
                pipe.vae.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
            pipe.transformer.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
        if self.config.taehv_vae_mode != "off":
            pipe.install_taehv_vae(
                mode=str(self.config.taehv_vae_mode),
                checkpoint=self.config.taehv_checkpoint,
            )
        self.state.pipe = pipe
        return pipe

    def reset(self) -> None:
        self.stop_continuous()
        self.state.wah_state = None
        self.state.prompt = ""
        self.state.current_pose = None
        self.state.session_id = ""
        self.state.generated_chunks = 0
        self.state.last_video_path = None
        self.state.history_items = []
        self.state.window_num_frames = 33
        with self.control_lock:
            self.continuous.playing_index = -1
            self.continuous.generating_chunk = None
            self.continuous.last_latency = None
            self.continuous.last_profile = None
            self.continuous.error = ""

    def _new_generator(self):
        import torch

        if str(self.config.device).startswith("cuda"):
            return torch.Generator(device=self.config.device).manual_seed(int(self.config.seed))
        return None

    def _init_generation(
        self,
        *,
        pipe: Any,
        prompt: str,
        image: Image.Image,
        translation_scale: float,
        chunks_per_generate: int,
    ) -> None:
        np = numpy_module()
        self.state.session_id = uuid.uuid4().hex[:12]
        self.state.prompt = prompt
        self.state.current_pose = np.eye(4, dtype=np.float32)
        self.state.generated_chunks = 0
        self.state.last_video_path = None
        self.state.history_items = []
        state = pipe.init_autoregressive_state(
            prompt=prompt,
            image=image,
            conditioning_type="camera",
            lora_path=self.config.lora_path,
            height=int(self.config.height),
            width=int(self.config.width),
            num_frames=max(1, int(chunks_per_generate)) * 33,
            generator=self._new_generator(),
            output_type="np",
            camera_control_translation_scale=float(translation_scale),
            camera_control_pi3x_keyframe_memory=bool(self.config.pi3x_keyframe_memory),
            camera_control_warp_render_mode=str(self.config.camera_warp_render_mode),
            camera_control_pi3_pixel_limit=max(int(self.config.camera_pi3_pixel_limit), 1),
            camera_control_mesh_samples_per_axis=max(int(self.config.camera_mesh_samples_per_axis), 1),
            warp_history_downsample_mode=str(self.config.warp_history_downsample_mode),
            visible_token_drop=bool(self.config.visible_token_drop),
            pyramid_num_inference_steps_list=list(self.config.pyramid_num_inference_steps_list)
            if self.config.pyramid_num_inference_steps_list is not None
            else None,
            is_amplify_first_chunk=bool(self.config.amplify_first_chunk),
        )
        if bool(self.config.visible_token_drop) and float(self.config.visible_token_threshold) != 0.1:
            state["attention_kwargs"] = {
                **(state.get("attention_kwargs") or {}),
                "history_visible_token_threshold": float(self.config.visible_token_threshold),
            }
        self.state.wah_state = state
        self.state.window_num_frames = int(state["window_num_frames"])

    def _generate_one_chunk_locked(
        self,
        *,
        pipe: Any,
        active: set[str],
        translation_scale: float,
        rotation_degrees: float,
    ) -> dict[str, Any]:
        from scripts.infer_warp_as_history import unwrap_video_frames

        if self.state.wah_state is None or self.state.current_pose is None:
            raise RuntimeError("Continuous generation state has not been initialized.")
        history_items = self.state.history_items
        if history_items is None:
            history_items = []
            self.state.history_items = history_items

        profile: dict[str, Any] = {
            "chunk": int(self.state.generated_chunks) + 1,
            "history_count_before": len(history_items),
        }
        profile_start = time.perf_counter()

        previous_ui_scale = float(self.state.wah_state.get("web_ui_translation_scale", translation_scale))
        if self.state.generated_chunks > 0:
            multiplier = self.state.wah_state.get("web_ui_translation_effective_multiplier")
            if multiplier is None:
                effective_scale = float(self.state.wah_state.get("camera_translation_effective_scale", previous_ui_scale))
                multiplier = effective_scale / previous_ui_scale if abs(previous_ui_scale) > 1.0e-8 else 1.0
                self.state.wah_state["web_ui_translation_effective_multiplier"] = float(multiplier)
            self.state.wah_state["camera_translation_effective_scale"] = float(translation_scale) * float(multiplier)
        self.state.wah_state["camera_control_translation_scale"] = float(translation_scale)
        self.state.wah_state["web_ui_translation_scale"] = float(translation_scale)

        start_chunk = int(self.state.generated_chunks)
        is_first_generated_chunk = self.state.generated_chunks == 0
        step_start = time.perf_counter()
        camera_chunk, end_pose = build_camera_chunk(
            self.state.current_pose,
            active=active,
            rotation_degrees=float(rotation_degrees),
            window_num_frames=int(self.state.window_num_frames),
            include_start=is_first_generated_chunk,
        )
        profile["camera_ms"] = round((time.perf_counter() - step_start) * 1000.0, 2)
        step_start = time.perf_counter()
        chunk_video, self.state.wah_state = pipe.generate_next_chunk(
            self.state.wah_state,
            camera_poses=camera_chunk,
            output_type="np",
        )
        profile["generate_ms"] = round((time.perf_counter() - step_start) * 1000.0, 2)
        if is_first_generated_chunk:
            effective_scale = float(self.state.wah_state.get("camera_translation_effective_scale", translation_scale))
            multiplier = effective_scale / float(translation_scale) if abs(float(translation_scale)) > 1.0e-8 else 1.0
            self.state.wah_state["web_ui_translation_effective_multiplier"] = float(multiplier)
        self.state.current_pose = end_pose
        self.state.generated_chunks += 1

        step_start = time.perf_counter()
        frames = unwrap_video_frames(chunk_video)
        profile["unwrap_ms"] = round((time.perf_counter() - step_start) * 1000.0, 2)
        if not frames:
            raise RuntimeError("No newly finalized frames were produced.")

        segment_name = f"{self.state.session_id}_{self.state.generated_chunks:04d}_segment.mp4"
        segment_path = self.config.output_dir / segment_name
        step_start = time.perf_counter()
        write_mse_segment(segment_path, frames, fps=int(self.config.fps))
        profile["mse_segment_ms"] = round((time.perf_counter() - step_start) * 1000.0, 2)
        self.state.last_video_path = segment_path
        rel_name = segment_path.name
        profile["frames"] = len(frames)
        profile["segment_bytes"] = int(segment_path.stat().st_size) if segment_path.exists() else 0
        item = {
            "index": len(history_items),
            "label": f"Chunks {start_chunk + 1}-{self.state.generated_chunks}",
            "video_url": f"/media/{rel_name}?v={int(time.time() * 1000)}",
            "segment_url": f"/media/{segment_name}?v={int(time.time() * 1000)}",
            "output_path": str(segment_path),
            "frames": len(frames),
            "preview_urls": [],
            "preview_fps": int(self.config.fps),
            "chunk_start": start_chunk + 1,
            "chunk_end": self.state.generated_chunks,
            "server_profile": profile,
        }
        history_items.append(item)
        profile["total_ms"] = round((time.perf_counter() - profile_start) * 1000.0, 2)
        print(json.dumps({"event": "web_chunk_profile", **profile}, ensure_ascii=True), flush=True)
        return item

    def start_continuous(
        self,
        *,
        prompt: str,
        image: Image.Image | None,
        active: set[str],
        translation_scale: float,
        rotation_degrees: float,
        target_buffer: int,
        reset: bool,
    ) -> dict[str, Any]:
        with self.lock:
            pipe = self.load_pipeline()
            if reset:
                self.reset()
            if self.state.wah_state is None:
                if image is None:
                    raise ValueError("Upload a first frame before starting.")
                if not prompt.strip():
                    raise ValueError("Prompt is required before starting.")
                self._init_generation(
                    pipe=pipe,
                    prompt=prompt.strip(),
                    image=image.convert("RGB"),
                    translation_scale=float(translation_scale),
                    chunks_per_generate=max(1, int(target_buffer) + 2),
                )
        with self.control_lock:
            if self.continuous.running:
                return self.status()
            self.continuous.active = set(active)
            self.continuous.translation_scale = float(translation_scale)
            self.continuous.rotation_degrees = float(rotation_degrees)
            self.continuous.target_buffer = max(0, int(target_buffer))
            self.continuous.error = ""
            self.continuous.stop_event = threading.Event()
            self.continuous.running = True
            worker = threading.Thread(target=self._continuous_loop, args=(pipe,), daemon=True)
            self.continuous.worker = worker
            worker.start()
        return self.status()

    def stop_continuous(self) -> None:
        worker = None
        with self.control_lock:
            if self.continuous.stop_event is not None:
                self.continuous.stop_event.set()
            worker = self.continuous.worker
            self.continuous.running = False
            self.continuous.worker = None
            self.continuous.stop_event = None
            self.continuous.generating_chunk = None
        if worker is not None and worker.is_alive() and worker is not threading.current_thread():
            worker.join(timeout=2.0)

    def update_control(
        self,
        *,
        active: set[str] | None = None,
        translation_scale: float | None = None,
        rotation_degrees: float | None = None,
        target_buffer: int | None = None,
        playing_index: int | None = None,
    ) -> dict[str, Any]:
        with self.control_lock:
            if active is not None:
                self.continuous.active = set(active)
            if translation_scale is not None:
                self.continuous.translation_scale = float(translation_scale)
            if rotation_degrees is not None:
                self.continuous.rotation_degrees = float(rotation_degrees)
            if target_buffer is not None:
                self.continuous.target_buffer = max(0, int(target_buffer))
            if playing_index is not None:
                self.continuous.playing_index = int(playing_index)
        return self.status()

    def _continuous_loop(self, pipe: Any) -> None:
        while True:
            with self.control_lock:
                stop_event = self.continuous.stop_event
                if not self.continuous.running or stop_event is None or stop_event.is_set():
                    break
                playing_index = int(self.continuous.playing_index)
                target_buffer = max(0, int(self.continuous.target_buffer))
                generation_buffer = max(1, target_buffer)
            with self.lock:
                produced = len(self.state.history_items or [])
            if produced - (playing_index + 1) >= generation_buffer:
                time.sleep(0.05)
                continue

            with self.control_lock:
                active = set(self.continuous.active or set())
                translation_scale = float(self.continuous.translation_scale)
                rotation_degrees = float(self.continuous.rotation_degrees)
                self.continuous.generating_chunk = int(self.state.generated_chunks) + 1

            start = time.perf_counter()
            try:
                with self.lock:
                    item = self._generate_one_chunk_locked(
                        pipe=pipe,
                        active=active,
                        translation_scale=translation_scale,
                        rotation_degrees=rotation_degrees,
                    )
                with self.control_lock:
                    self.continuous.last_latency = time.perf_counter() - start
                    self.continuous.last_profile = dict(item.get("server_profile") or {})
                    self.continuous.generating_chunk = None
            except Exception as exc:
                traceback.print_exc()
                with self.control_lock:
                    self.continuous.error = str(exc)
                    self.continuous.running = False
                    if self.continuous.stop_event is not None:
                        self.continuous.stop_event.set()
                    self.continuous.generating_chunk = None
                break

    def status(self) -> dict[str, Any]:
        # Keep status/chunk polling independent from the long generation lock.
        # The generation thread appends fully-built history items atomically; UI
        # polling should not wait for the next DiT chunk before it can see the
        # previous one.
        history = list(self.state.history_items or [])
        chunks_generated = int(self.state.generated_chunks)
        window_num_frames = int(self.state.window_num_frames)
        with self.control_lock:
            queued = len(history) - (int(self.continuous.playing_index) + 1)
            return {
                "ok": True,
                "running": bool(self.continuous.running),
                "chunks_generated": chunks_generated,
                "window_num_frames": window_num_frames,
                "history_count": len(history),
                "queued_chunks": max(0, queued),
                "playing_index": int(self.continuous.playing_index),
                "target_buffer": int(self.continuous.target_buffer),
                "generating_chunk": self.continuous.generating_chunk,
                "last_latency": self.continuous.last_latency,
                "last_profile": self.continuous.last_profile,
                "error": self.continuous.error,
            }

    def chunks_since(self, since: int) -> dict[str, Any]:
        history = [
            item
            for item in list(self.state.history_items or [])
            if int(item.get("index", -1)) > int(since)
        ]
        payload = self.status()
        payload["chunks"] = history
        return payload

HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Warp-as-History Realtime Demo</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #080a0c;
      --panel: #11161a;
      --panel2: #171e23;
      --line: #2d3940;
      --text: #eef5f4;
      --muted: #93a4aa;
      --accent: #5df0bf;
      --error: #ff7474;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.4 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      overflow: hidden;
    }
    button, input, textarea { font: inherit; }
    button {
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel2);
      color: var(--text);
      cursor: pointer;
    }
    button:hover, button.active { border-color: var(--accent); }
    button.active { background: color-mix(in srgb, var(--accent) 22%, var(--panel2)); }
    button:disabled { opacity: .42; cursor: not-allowed; }
    label { display: block; margin: 10px 0 5px; color: var(--muted); font-size: 12px; }
    textarea, input[type="file"], input[type="number"] {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel2);
      color: var(--text);
      padding: 8px;
    }
    textarea { min-height: 92px; resize: vertical; }
    input[type="range"] { width: 100%; }
    .app {
      width: 100vw;
      height: 100vh;
      height: 100dvh;
      display: grid;
      grid-template-columns: 300px minmax(0, 1fr);
      gap: 12px;
      padding: 12px;
      overflow: hidden;
    }
    .panel, .topbar, .stats {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: var(--panel);
    }
    .panel { min-width: 0; min-height: 0; padding: 12px; overflow-y: auto; }
    .title { margin: 0 0 12px; font-size: 18px; color: var(--accent); }
    .button-row { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 10px; }
    #start { grid-column: 1 / -1; color: #06110d; background: var(--accent); border-color: var(--accent); font-weight: 800; }
    #reset { border-color: color-mix(in srgb, var(--error) 70%, var(--line)); }
    .pad-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 7px; margin: 6px 0 14px; }
    .pad-grid span { min-height: 36px; }
    .row { display: grid; grid-template-columns: minmax(0, 1fr) 70px; gap: 8px; align-items: center; }
    .stage {
      min-width: 0;
      min-height: 0;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      gap: 10px;
      overflow: hidden;
    }
    .topbar { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 12px; align-items: center; padding: 10px 12px; }
    .status { overflow-wrap: anywhere; }
    .keys { display: flex; gap: 5px; flex-wrap: wrap; justify-content: end; }
    .key {
      min-width: 26px;
      height: 24px;
      display: grid;
      place-items: center;
      border: 1px solid var(--line);
      border-radius: 6px;
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
    }
    .key.active { color: #06110d; background: var(--accent); border-color: var(--accent); }
    .screen {
      position: relative;
      width: 100%;
      height: 100%;
      min-height: 0;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #020304;
      overflow: hidden;
    }
    video { width: 100%; height: 100%; object-fit: contain; display: block; background: #020304; }
    .clock, .log {
      position: absolute;
      z-index: 2;
      border: 1px solid rgba(93, 240, 191, .34);
      border-radius: 8px;
      background: rgba(5, 8, 9, .76);
      backdrop-filter: blur(10px);
      pointer-events: none;
    }
    .clock { left: 14px; top: 14px; padding: 8px 10px; }
    .clock strong { display: block; color: var(--accent); font-size: 26px; line-height: 1; font-variant-numeric: tabular-nums; }
    .clock span { color: var(--muted); font-size: 11px; }
    .log {
      left: 14px;
      bottom: 14px;
      width: min(430px, calc(100% - 28px));
      max-height: 128px;
      overflow: hidden;
      padding: 8px 10px;
      font-size: 11px;
      font-variant-numeric: tabular-nums;
    }
    .log-title { margin-bottom: 4px; color: var(--accent); font-weight: 850; text-transform: uppercase; letter-spacing: .08em; }
    .log-line { overflow: hidden; white-space: nowrap; text-overflow: ellipsis; color: #d8e5e4; opacity: .82; }
    .log-line:first-child { color: #fff; opacity: 1; }
    .stats {
      min-height: 0;
      display: grid;
      grid-template-columns: repeat(5, minmax(72px, .7fr)) minmax(160px, 1.5fr);
      gap: 10px;
      padding: 8px 10px;
      overflow: hidden;
    }
    .metric span { display: block; color: var(--muted); font-size: 11px; }
    .metric strong { color: var(--accent); font-variant-numeric: tabular-nums; }
    .history { overflow: hidden; white-space: nowrap; text-overflow: ellipsis; color: var(--muted); }
    @media (max-width: 900px) {
      body { overflow: auto; }
      .app { width: auto; height: auto; min-height: 100vh; grid-template-columns: 1fr; overflow: visible; }
      .screen { height: 60vh; min-height: 320px; }
      .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <main class="app">
    <aside class="panel">
      <h1 class="title">Warp-as-History Realtime</h1>
      <label for="firstFrame">First Frame</label>
      <input id="firstFrame" type="file" accept="image/*" />
      <label for="prompt">Prompt</label>
      <textarea id="prompt" spellcheck="false">A realistic scene with camera movemnet.</textarea>
      <div class="button-row">
        <button id="start" type="button">Start</button>
        <button id="stop" type="button">Stop</button>
        <button id="reset" type="button">Reset</button>
      </div>

      <label>Move</label>
      <div class="pad-grid">
        <span></span><button class="toggle" data-control="forward" type="button">W</button><span></span>
        <button class="toggle" data-control="strafe_left" type="button">A</button>
        <button class="toggle" data-control="backward" type="button">S</button>
        <button class="toggle" data-control="strafe_right" type="button">D</button>
        <button class="toggle" data-control="descend" type="button">Q</button>
        <span></span><button class="toggle" data-control="rise" type="button">E</button>
      </div>

      <label>Rotate</label>
      <div class="pad-grid">
        <span></span><button class="toggle" data-control="pitch_up" type="button">I</button><span></span>
        <button class="toggle" data-control="yaw_left" type="button">J</button>
        <button class="toggle" data-control="pitch_down" type="button">K</button>
        <button class="toggle" data-control="yaw_right" type="button">L</button>
        <button class="toggle" data-control="roll_left" type="button">U</button>
        <span></span><button class="toggle" data-control="roll_right" type="button">O</button>
      </div>

      <label>Translation Scale</label>
      <div class="row">
        <input id="translationScale" type="range" min="0" max="0.5" step="0.005" value="0.1" />
        <input id="translationScaleValue" type="number" min="0" max="10" step="0.005" value="0.1" />
      </div>
      <label>Rotation Degrees</label>
      <div class="row">
        <input id="rotationDegrees" type="range" min="0" max="30" step="0.5" value="4" />
        <input id="rotationDegreesValue" type="number" min="0" max="180" step="0.5" value="4" />
      </div>
      <label>Target Buffer</label>
      <div class="row">
        <input id="targetBuffer" type="range" min="0" max="4" step="1" value="2" />
        <input id="targetBufferValue" type="number" min="0" max="8" step="1" value="2" />
      </div>
    </aside>

    <section class="stage">
      <div class="topbar">
        <div id="status" class="status">Idle</div>
        <div class="keys">
          <span class="key" data-keycap="forward">W</span>
          <span class="key" data-keycap="strafe_left">A</span>
          <span class="key" data-keycap="backward">S</span>
          <span class="key" data-keycap="strafe_right">D</span>
          <span class="key" data-keycap="yaw_left">J</span>
          <span class="key" data-keycap="pitch_up">I</span>
          <span class="key" data-keycap="pitch_down">K</span>
          <span class="key" data-keycap="yaw_right">L</span>
        </div>
      </div>
      <div class="screen">
        <video id="video" autoplay muted playsinline preload="auto"></video>
        <div class="clock"><strong id="elapsed">00:00.0</strong><span id="clockDetail">waiting</span></div>
        <div class="log"><div class="log-title">Event Log</div><div id="eventLog"></div></div>
      </div>
      <div class="stats">
        <div class="metric"><span>Chunks</span><strong id="chunks">0</strong></div>
        <div class="metric"><span>Playing</span><strong id="playing">-</strong></div>
        <div class="metric"><span>Buffer</span><strong id="buffer">0</strong></div>
        <div class="metric"><span>Latency</span><strong id="latency">-</strong></div>
        <div class="metric"><span>Speed</span><strong id="speed">-</strong></div>
        <div class="history" id="history">History: -</div>
      </div>
    </section>
  </main>

  <script>
    const APP_FPS = __FPS__;
    const $ = id => document.getElementById(id);
    const active = new Set();
    const keyMap = {
      KeyW: 'forward', ArrowUp: 'forward',
      KeyS: 'backward', ArrowDown: 'backward',
      KeyA: 'strafe_left', KeyD: 'strafe_right',
      KeyQ: 'descend', KeyE: 'rise',
      KeyJ: 'yaw_left', ArrowLeft: 'yaw_left',
      KeyL: 'yaw_right', ArrowRight: 'yaw_right',
      KeyI: 'pitch_up', KeyK: 'pitch_down',
      KeyU: 'roll_left', KeyO: 'roll_right',
    };
    const controls = {
      translationScale: $('translationScale'),
      translationScaleValue: $('translationScaleValue'),
      rotationDegrees: $('rotationDegrees'),
      rotationDegreesValue: $('rotationDegreesValue'),
      targetBuffer: $('targetBuffer'),
      targetBufferValue: $('targetBufferValue'),
    };
    let running = false;
    let startedAt = 0;
    let statusPayload = null;
    let mediaSource = null;
    let sourceBuffer = null;
    let queue = [];
    let appending = false;
    let seen = new Set();
    let pending = new Set();
    let lastChunkIndex = -1;
    let playingIndex = -1;
    let logs = [];
    let startBusy = false;
    let stopBusy = false;
    let resetBusy = false;

    function escapeHtml(text) {
      return String(text).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    }
    function log(message) {
      const t = new Date().toLocaleTimeString('en-US', { hour12: false });
      logs.unshift(`${t} ${message}`);
      logs = logs.slice(0, 7);
      $('eventLog').innerHTML = logs.map(line => `<div class="log-line">${escapeHtml(line)}</div>`).join('');
    }
    function postJSON(url, data) {
      return fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data || {}) }).then(r => r.json());
    }
    function controlPayload(extra) {
      return Object.assign({
        active: [...active],
        translation_scale: Number(controls.translationScale.value),
        rotation_degrees: Number(controls.rotationDegrees.value),
        target_buffer: Number(controls.targetBuffer.value),
        playing_index: playingIndex,
      }, extra || {});
    }
    function renderActive() {
      document.querySelectorAll('[data-control]').forEach(el => el.classList.toggle('active', active.has(el.dataset.control)));
      document.querySelectorAll('[data-keycap]').forEach(el => el.classList.toggle('active', active.has(el.dataset.keycap)));
    }
    function updateButtons() {
      $('start').disabled = running || startBusy || resetBusy;
      $('stop').disabled = (!running && !startBusy) || stopBusy || resetBusy;
      $('reset').disabled = resetBusy;
    }
    let controlTimer = null;
    function sendControl() {
      renderActive();
      if (!running) return;
      clearTimeout(controlTimer);
      controlTimer = setTimeout(() => {
        postJSON('/api/control', controlPayload()).catch(error => log(`control error: ${error.message}`));
      }, 40);
    }
    function syncNumber(name) {
      const range = controls[name];
      const value = controls[`${name}Value`];
      range.addEventListener('input', () => { value.value = range.value; sendControl(); });
      range.addEventListener('pointerup', () => range.blur());
      range.addEventListener('keyup', event => {
        if (event.key === 'Enter' || event.key === 'Escape') range.blur();
      });
      value.addEventListener('change', () => { range.value = value.value; sendControl(); value.blur(); });
    }

    function isTextEntryTarget(target) {
      if (!target || !target.matches) return false;
      if (target.matches('textarea')) return true;
      if (!target.matches('input')) return false;
      return ['text', 'number', 'search', 'email', 'url', 'password', 'file'].includes(String(target.type || 'text'));
    }

    document.querySelectorAll('.toggle').forEach(button => {
      button.addEventListener('click', () => {
        const name = button.dataset.control;
        active.has(name) ? active.delete(name) : active.add(name);
        sendControl();
      });
    });
    document.addEventListener('keydown', event => {
      if (event.repeat || isTextEntryTarget(event.target)) return;
      const name = keyMap[event.code];
      if (!name) return;
      event.preventDefault();
      active.add(name);
      sendControl();
    });
    document.addEventListener('keyup', event => {
      if (isTextEntryTarget(event.target)) return;
      const name = keyMap[event.code];
      if (!name) return;
      event.preventDefault();
      active.delete(name);
      sendControl();
    });
    syncNumber('translationScale');
    syncNumber('rotationDegrees');
    syncNumber('targetBuffer');

    function supportedMime() {
      for (const type of [
        'video/mp4; codecs="avc1.64001f"',
        'video/mp4; codecs="avc1.4d401f"',
        'video/mp4; codecs="avc1.42E01E"',
        'video/mp4',
      ]) {
        if (window.MediaSource && MediaSource.isTypeSupported(type)) return type;
      }
      return '';
    }
    function resetPlayer() {
      queue = [];
      appending = false;
      seen = new Set();
      pending = new Set();
      lastChunkIndex = -1;
      playingIndex = -1;
      if (mediaSource && mediaSource.readyState === 'open') {
        try { mediaSource.endOfStream(); } catch (_) {}
      }
      if ($('video').src) URL.revokeObjectURL($('video').src);
      mediaSource = null;
      sourceBuffer = null;
      $('video').removeAttribute('src');
      $('video').load();
      updateHistory();
    }
    async function initPlayer() {
      resetPlayer();
      const type = supportedMime();
      if (!type) throw new Error('MediaSource MP4 is not supported');
      mediaSource = new MediaSource();
      $('video').src = URL.createObjectURL(mediaSource);
      await new Promise((resolve, reject) => {
        mediaSource.addEventListener('sourceopen', () => {
          try {
            sourceBuffer = mediaSource.addSourceBuffer(type);
            sourceBuffer.mode = 'sequence';
            sourceBuffer.addEventListener('updateend', pump);
            resolve();
          } catch (error) {
            reject(error);
          }
        }, { once: true });
      });
    }
    async function fetchChunk(item) {
      if (!sourceBuffer || seen.has(item.index) || pending.has(item.index)) return;
      const url = item.segment_url || item.video_url || item.url;
      if (!url) {
        log(`missing video url for chunk ${item.index + 1}`);
        lastChunkIndex = Math.max(lastChunkIndex, item.index);
        return;
      }
      pending.add(item.index);
      log(`fetch chunk ${item.index + 1}`);
      try {
        const response = await fetch(url, { cache: 'no-store' });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        queue.push({ item, data: await response.arrayBuffer() });
        pump();
      } catch (error) {
        pending.delete(item.index);
        log(`fetch error ${item.index + 1}: ${error.message}`);
      }
    }
    function pump() {
      if (appending || !sourceBuffer || sourceBuffer.updating || !queue.length) return;
      const { item, data } = queue.shift();
      appending = true;
      const done = () => {
        sourceBuffer.removeEventListener('error', fail);
        sourceBuffer.removeEventListener('updateend', done);
        appending = false;
        seen.add(item.index);
        pending.delete(item.index);
        lastChunkIndex = Math.max(lastChunkIndex, item.index);
        if (playingIndex < 0) playingIndex = item.index;
        $('video').play().catch(() => {});
        log(`append chunk ${item.index + 1}`);
        updateHistory();
        sendControl();
        pump();
      };
      const fail = () => {
        sourceBuffer.removeEventListener('error', fail);
        sourceBuffer.removeEventListener('updateend', done);
        appending = false;
        log(`append error chunk ${item.index + 1}`);
        pump();
      };
      sourceBuffer.addEventListener('updateend', done, { once: true });
      sourceBuffer.addEventListener('error', fail, { once: true });
      sourceBuffer.appendBuffer(data);
    }
    $('video').addEventListener('timeupdate', () => {
      if (!statusPayload || !statusPayload.window_num_frames || lastChunkIndex < 0) return;
      const chunkSeconds = Math.max(0.001, Number(statusPayload.window_num_frames) / APP_FPS);
      const index = Math.min(lastChunkIndex, Math.max(0, Math.floor($('video').currentTime / chunkSeconds)));
      if (index !== playingIndex) {
        playingIndex = index;
        log(`display chunk ${index + 1}`);
        updateHistory();
        sendControl();
      }
    });
    function updateHistory() {
      const items = [...seen].sort((a, b) => a - b).slice(-16).map(index => index === playingIndex ? `[${index + 1}]` : String(index + 1));
      $('history').textContent = `History: ${items.length ? items.join(' ') : '-'}`;
      $('playing').textContent = playingIndex >= 0 ? String(playingIndex + 1) : '-';
    }
    async function pollStatus() {
      try {
        const data = await fetch('/api/status', { cache: 'no-store' }).then(r => r.json());
        statusPayload = data;
        running = !!data.running;
        updateButtons();
        const text = data.error ? `Error: ${data.error}` : data.running ? `Running${data.generating_chunk !== null ? ` · generating ${data.generating_chunk}` : ''}` : 'Idle';
        if ($('status').textContent !== text) log(text);
        $('status').textContent = text;
        $('chunks').textContent = data.chunks_generated ?? 0;
        $('buffer').textContent = data.queued_chunks ?? 0;
        $('latency').textContent = data.last_latency ? `${data.last_latency.toFixed(2)}s` : '-';
        $('speed').textContent = data.last_latency && data.window_num_frames ? `${((data.window_num_frames / APP_FPS) / data.last_latency).toFixed(2)}x` : '-';
      } catch (error) {
        $('status').textContent = `Status error: ${error.message}`;
      }
    }
    async function pollChunks() {
      if (!running && lastChunkIndex >= 0) return;
      try {
        const data = await fetch(`/api/chunks?since=${lastChunkIndex}`, { cache: 'no-store' }).then(r => r.json());
        if (data.chunks && data.chunks.length) log(`received ${data.chunks.length} chunk(s)`);
        for (const item of data.chunks || []) fetchChunk(item);
      } catch (error) {
        log(`chunk poll error: ${error.message}`);
      }
    }
    $('start').addEventListener('click', async () => {
      if (running || startBusy) return;
      startBusy = true;
      updateButtons();
      try {
        log('starting');
        await initPlayer();
        const form = new FormData();
        form.append('prompt', $('prompt').value || '');
        form.append('translation_scale', String(controls.translationScale.value));
        form.append('rotation_degrees', String(controls.rotationDegrees.value));
        form.append('target_buffer', String(controls.targetBuffer.value));
        form.append('active', JSON.stringify([...active]));
        if ($('firstFrame').files[0]) form.append('first_frame', $('firstFrame').files[0]);
        const data = await fetch('/api/start', { method: 'POST', body: form }).then(r => r.json());
        if (!data.ok) throw new Error(data.error || 'start failed');
        running = true;
        startedAt = performance.now();
        log('worker started');
      } catch (error) {
        log(`start error: ${error.message}`);
        $('status').textContent = `Start error: ${error.message}`;
      } finally {
        startBusy = false;
        updateButtons();
      }
    });
    $('stop').addEventListener('click', async () => {
      if (stopBusy || (!running && !startBusy)) return;
      stopBusy = true;
      updateButtons();
      try {
        log('stopping');
        await postJSON('/api/stop', {});
        running = false;
        log('stopped');
        pollStatus();
      } finally {
        stopBusy = false;
        updateButtons();
      }
    });
    $('reset').addEventListener('click', async () => {
      if (resetBusy) return;
      resetBusy = true;
      updateButtons();
      try {
        if (running || startBusy) {
          log('stopping before reset');
          await postJSON('/api/stop', {});
        }
        await postJSON('/api/reset', {});
        running = false;
        startedAt = 0;
        active.clear();
        renderActive();
        resetPlayer();
        log('reset');
        pollStatus();
      } finally {
        resetBusy = false;
        updateButtons();
      }
    });
    function tick() {
      if (running && startedAt) {
        const elapsed = (performance.now() - startedAt) / 1000;
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed - minutes * 60;
        $('elapsed').textContent = `${String(minutes).padStart(2, '0')}:${seconds.toFixed(1).padStart(4, '0')}`;
        $('clockDetail').textContent = `${new Date().toLocaleTimeString('en-US', { hour12: false })} · chunk ${playingIndex >= 0 ? playingIndex + 1 : '-'}`;
      } else {
        $('elapsed').textContent = '00:00.0';
        $('clockDetail').textContent = 'waiting';
      }
      requestAnimationFrame(tick);
    }
    log('idle');
    renderActive();
    updateButtons();
    setInterval(pollStatus, 500);
    setInterval(pollChunks, 220);
    pollStatus();
    tick();
  </script>
</body>
</html>"""


def read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or 0)
    if length <= 0:
        return {}
    return json.loads(handler.rfile.read(length).decode("utf-8"))


class MiniForm:
    def __init__(self, fields: dict[str, str]):
        self.fields = fields

    def getfirst(self, name: str, default: str = "") -> str:
        return self.fields.get(name, default)


def parse_multipart_form(handler: BaseHTTPRequestHandler) -> tuple[MiniForm, Image.Image | None]:
    content_type = handler.headers.get("Content-Type", "")
    length = int(handler.headers.get("Content-Length") or 0)
    body = handler.rfile.read(length) if length > 0 else b""
    if not content_type.startswith("multipart/form-data"):
        return MiniForm({}), None
    raw = b"Content-Type: " + content_type.encode("utf-8") + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
    message = BytesParser(policy=policy.default).parsebytes(raw)
    fields: dict[str, str] = {}
    image: Image.Image | None = None
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        payload = part.get_payload(decode=True) or b""
        filename = part.get_filename()
        if name == "first_frame" and filename and payload:
            from PIL import Image

            image = Image.open(io.BytesIO(payload)).convert("RGB")
        elif not filename:
            fields[str(name)] = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
    return MiniForm(fields), image


def form_image_and_controls(handler: BaseHTTPRequestHandler) -> tuple[MiniForm, Image.Image | None, set[str]]:
    form, image = parse_multipart_form(handler)
    try:
        active = {str(item) for item in json.loads(form.getfirst("active", "[]"))}
    except Exception:
        active = set()
    return form, image, active


class LazyRealtimeApp:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output_dir = selected_output_dir(args)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._app: Any | None = None

    def _ensure_app(self) -> Any:
        if self._app is not None:
            return self._app
        from scripts.infer_warp_as_history import disable_diffusers_optional_attention

        patch_diffusers_kernel_attr_resolution()
        preload_cached_flash_attn3_hub_kernel()
        if use_optional_attention(self.args) and not self.args.enable_compile:
            from helios.modules.helios_kernels import replace_rope_with_flash_rope

            replace_rope_with_flash_rope()
        else:
            disable_diffusers_optional_attention()

        self._app = WarpControlApp(build_config(self.args))
        return self._app

    def load_pipeline(self) -> Any:
        return self._ensure_app().load_pipeline()

    def reset(self) -> None:
        if self._app is None:
            return None
        return self._ensure_app().reset()

    def stop_continuous(self) -> None:
        if self._app is None:
            return None
        return self._ensure_app().stop_continuous()

    def status(self) -> dict[str, Any]:
        if self._app is None:
            return {
                "ok": True,
                "running": False,
                "chunks_generated": 0,
                "window_num_frames": 33,
                "history_count": 0,
                "queued_chunks": 0,
                "playing_index": -1,
                "target_buffer": 2,
                "generating_chunk": None,
                "last_latency": None,
                "last_profile": None,
                "error": "",
            }
        return self._app.status()

    def chunks_since(self, since: int) -> dict[str, Any]:
        if self._app is None:
            return {"ok": True, "chunks": [], "latest_index": -1}
        return self._app.chunks_since(since)

    def start_continuous(self, **kwargs: Any) -> dict[str, Any]:
        return self._ensure_app().start_continuous(**kwargs)

    def update_control(self, **kwargs: Any) -> dict[str, Any]:
        return self._ensure_app().update_control(**kwargs)


def make_handler(app: LazyRealtimeApp, *, fps: int):
    class Handler(BaseHTTPRequestHandler):
        server_version = "WarpRealtimeHTTP/0.2"

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                text_response(self, HTML_PAGE.replace("__FPS__", str(int(fps))))
                return
            if parsed.path == "/api/status":
                json_response(self, app.status())
                return
            if parsed.path == "/api/chunks":
                query = parse_qs(parsed.query)
                since = int(query.get("since", ["-1"])[0])
                json_response(self, app.chunks_since(since))
                return
            if parsed.path.startswith("/media/"):
                self._serve_media(parsed.path)
                return
            text_response(self, "Not found", status=HTTPStatus.NOT_FOUND, content_type="text/plain; charset=utf-8")

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/api/reset":
                app.reset()
                json_response(self, {"ok": True, "message": "State reset"})
                return
            if parsed.path == "/api/start":
                self._handle_start()
                return
            if parsed.path == "/api/stop":
                app.stop_continuous()
                json_response(self, app.status())
                return
            if parsed.path == "/api/control":
                self._handle_control()
                return
            json_response(self, {"ok": False, "error": "Not found"}, status=HTTPStatus.NOT_FOUND)

        def _serve_media(self, path: str) -> None:
            rel = unquote(path[len("/media/") :]).split("?", 1)[0]
            if not rel or "/" in rel or "\\" in rel:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            media_path = (app.output_dir / rel).resolve()
            if not media_path.is_file() or app.output_dir not in media_path.parents:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            data = media_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _handle_start(self) -> None:
            try:
                form, image, active = form_image_and_controls(self)
                payload = app.start_continuous(
                    prompt=form.getfirst("prompt", ""),
                    image=image,
                    active=active,
                    translation_scale=float(form.getfirst("translation_scale", "0.1")),
                    rotation_degrees=float(form.getfirst("rotation_degrees", "4")),
                    target_buffer=int(float(form.getfirst("target_buffer", "2"))),
                    reset=form.getfirst("reset", "0") == "1",
                )
                json_response(self, payload)
            except Exception as exc:
                json_response(self, {"ok": False, "error": html.escape(str(exc))}, status=HTTPStatus.BAD_REQUEST)

        def _handle_control(self) -> None:
            try:
                data = read_json(self)
                active = data.get("active")
                payload = app.update_control(
                    active={str(item) for item in active} if active is not None else None,
                    translation_scale=data.get("translation_scale"),
                    rotation_degrees=data.get("rotation_degrees"),
                    target_buffer=data.get("target_buffer"),
                    playing_index=data.get("playing_index"),
                )
                json_response(self, payload)
            except Exception as exc:
                json_response(self, {"ok": False, "error": html.escape(str(exc))}, status=HTTPStatus.BAD_REQUEST)

    return Handler


def is_efficient_realtime(args: argparse.Namespace) -> bool:
    return str(args.preset) == "efficient_realtime"


def selected_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    return (DEFAULT_EFFICIENT_OUTPUT_DIR if is_efficient_realtime(args) else DEFAULT_OUTPUT_DIR).resolve()


def use_optional_attention(args: argparse.Namespace) -> bool:
    return bool(args.enable_optional_attention or is_efficient_realtime(args))


def build_config(args: argparse.Namespace) -> Any:
    from scripts.infer_warp_as_history import resolve_lora_path, resolve_model_path

    efficient = is_efficient_realtime(args)
    lora_path = args.lora_path
    if efficient and not args.no_lora and str(lora_path) == DEFAULT_WAH_LORA:
        lora_path = DEFAULT_EFFICIENT_WAH_LORA
    pyramid_steps = [1, 1, 1] if efficient else None
    if args.pyramid_steps:
        pyramid_steps = parse_pyramid_steps(args.pyramid_steps)
    downsample_mode = args.warp_history_downsample_mode or ("patch_mid" if efficient else "short")
    matmul_precision = str(args.matmul_precision or ("high" if efficient else ""))
    taehv_vae_mode = str(args.taehv_vae_mode)
    taehv_checkpoint = Path(args.taehv_checkpoint).expanduser().resolve() if str(args.taehv_vae_mode) != "off" else None

    return AppConfig(
        model_path=resolve_model_path(args.model_path),
        lora_path=None if args.no_lora else resolve_lora_path(lora_path),
        height=int(args.height),
        width=int(args.width),
        fps=int(args.fps),
        seed=int(args.seed),
        device=str(args.device),
        dtype=str(args.dtype),
        output_dir=selected_output_dir(args),
        enable_compile=bool(args.enable_compile),
        matmul_precision=matmul_precision,
        disable_progress=bool(args.disable_progress or efficient),
        pyramid_num_inference_steps_list=pyramid_steps,
        amplify_first_chunk=False if efficient else not bool(args.no_amplify_first_chunk),
        warp_history_downsample_mode=str(downsample_mode),
        visible_token_drop=not bool(args.no_visible_token_drop),
        visible_token_threshold=0.6 if efficient else float(args.visible_token_threshold),
        pi3x_keyframe_memory=not bool(args.no_pi3x_keyframe_memory),
        camera_warp_render_mode="target_fill" if efficient else str(args.camera_warp_render_mode),
        camera_pi3_pixel_limit=130000 if efficient else max(int(args.camera_pi3_pixel_limit), 1),
        camera_mesh_samples_per_axis=2 if efficient else max(int(args.camera_mesh_samples_per_axis), 1),
        taehv_vae_mode="full" if efficient else taehv_vae_mode,
        taehv_checkpoint=Path(REPO_ROOT / "checkpoints" / "taehv" / "taew2_1.pth").resolve()
        if efficient
        else taehv_checkpoint,
        enable_official_kernels=bool(args.enable_official_kernels or efficient),
    )


def main() -> None:
    args = parse_args()
    app = LazyRealtimeApp(args)
    if args.preload or is_efficient_realtime(args):
        app.load_pipeline()
    server = ThreadingHTTPServer((str(args.host), int(args.port)), make_handler(app, fps=int(args.fps)))
    config = build_config(args)
    print(
        json.dumps(
            {
                "event": "web_realtime_demo_ready",
                "url": f"http://{args.host}:{args.port}",
                "output_dir": str(app.output_dir),
                "preset": str(args.preset),
                "model_path": config.model_path,
                "lora_path": config.lora_path,
                "compile": bool(config.enable_compile),
                "fps": int(args.fps),
                "official_kernels": bool(config.enable_official_kernels),
                "taehv_vae_mode": str(config.taehv_vae_mode),
                "warp_history_downsample_mode": str(config.warp_history_downsample_mode),
                "pyramid_num_inference_steps_list": config.pyramid_num_inference_steps_list,
                "amplify_first_chunk": bool(config.amplify_first_chunk),
                "camera_warp_render_mode": str(config.camera_warp_render_mode),
                "camera_pi3_pixel_limit": int(config.camera_pi3_pixel_limit),
                "camera_mesh_samples_per_axis": int(config.camera_mesh_samples_per_axis),
                "visible_token_threshold": float(config.visible_token_threshold),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
