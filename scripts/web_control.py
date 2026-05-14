#!/usr/bin/env python3
from __future__ import annotations

import argparse
import cgi
import html
import io
import json
import math
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import imageio.v2 as imageio
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.infer_warp_as_history import (  # noqa: E402
    DEFAULT_MODEL,
    disable_diffusers_optional_attention,
    frame_to_uint8,
    resolve_model_path,
    torch_dtype_from_arg,
    unwrap_video_frames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive Warp-as-History camera-control web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--lora_path", default="")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "runs" / "web_control")
    parser.add_argument("--preload", action="store_true", help="Load the model before serving requests.")
    parser.add_argument(
        "--enable_optional_attention",
        action="store_true",
        help="Let diffusers import optional attention packages. Defaults to native PyTorch attention.",
    )
    return parser.parse_args()


def _json_response(handler: BaseHTTPRequestHandler, payload: dict[str, Any], status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _text_response(
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


def _rotation_x(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


def _rotation_y(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def _rotation_z(angle: float) -> np.ndarray:
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
    delta = np.eye(4, dtype=np.float32)
    delta[:3, :3] = _rotation_z(roll) @ _rotation_y(yaw) @ _rotation_x(pitch)
    delta[:3, 3] = translation.astype(np.float32, copy=False)
    return delta


def _control_delta(active: set[str], rotation_degrees: float) -> tuple[np.ndarray, float, float, float]:
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


def write_video(path: Path, frames: list[Any], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=int(fps), codec="libx264", macro_block_size=1) as writer:
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


@dataclass
class SessionState:
    pipe: Any = None
    wah_state: dict[str, Any] | None = None
    prompt: str = ""
    current_pose: np.ndarray | None = None
    session_id: str = ""
    generated_chunks: int = 0
    last_video_path: Path | None = None
    last_warp_path: Path | None = None
    history_items: list[dict[str, Any]] | None = None
    window_num_frames: int = 33


class WarpControlApp:
    def __init__(self, config: AppConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.state = SessionState()
        self.lock = threading.Lock()

    def load_pipeline(self) -> Any:
        if self.state.pipe is not None:
            return self.state.pipe
        import torch
        from warp_as_history import WarpAsHistoryPipeline

        dtype = torch_dtype_from_arg(self.config.dtype, self.config.device)
        pipe = WarpAsHistoryPipeline.from_pretrained(self.config.model_path, torch_dtype=dtype).to(self.config.device)
        self.state.pipe = pipe
        return pipe

    def reset(self) -> None:
        self.state.wah_state = None
        self.state.prompt = ""
        self.state.current_pose = None
        self.state.session_id = ""
        self.state.generated_chunks = 0
        self.state.last_video_path = None
        self.state.last_warp_path = None
        self.state.history_items = []
        self.state.window_num_frames = 33

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
        self.state.session_id = uuid.uuid4().hex[:12]
        self.state.prompt = prompt
        self.state.current_pose = np.eye(4, dtype=np.float32)
        self.state.generated_chunks = 0
        self.state.last_video_path = None
        self.state.last_warp_path = None
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
        )
        self.state.wah_state = state
        self.state.window_num_frames = int(state["window_num_frames"])

    def _warp_tensor_to_frames(self, tensor: Any) -> list[np.ndarray]:
        import torch

        if not torch.is_tensor(tensor):
            tensor = torch.as_tensor(tensor)
        video = tensor.detach().float().cpu()
        if video.ndim != 5:
            raise ValueError(f"Expected warp tensor [B,C,T,H,W], got {tuple(video.shape)}")
        video = video[0, :3].permute(1, 2, 3, 0)
        if video.numel() and float(video.min()) < 0.0:
            video = video / 2.0 + 0.5
        video = video.clamp(0.0, 1.0)
        array = (video.numpy() * 255.0).round().astype(np.uint8)
        return [array[i] for i in range(array.shape[0])]

    def _write_debug_warp_video(self, frame_start: int, frame_count: int) -> tuple[Path | None, int]:
        if self.state.wah_state is None:
            return None, 0
        rendered_chunks = self.state.wah_state.get("camera_warp_chunks", {})
        if not isinstance(rendered_chunks, dict) or not rendered_chunks:
            return None, 0

        frames: list[np.ndarray] = []
        for chunk_index in sorted(int(key) for key in rendered_chunks.keys()):
            rendered = rendered_chunks[chunk_index]
            if not isinstance(rendered, dict) or "warp_video" not in rendered:
                continue
            chunk_frames = self._warp_tensor_to_frames(rendered["warp_video"])
            if chunk_index > 0 and len(chunk_frames) > 1:
                chunk_frames = chunk_frames[1:]
            frames.extend(chunk_frames)

        if frame_start > 0:
            frames = frames[int(frame_start) :]
        if frame_count > 0:
            frames = frames[: int(frame_count)]
        if not frames:
            return None, 0

        filename = f"{self.state.session_id}_{self.state.generated_chunks:04d}_new_warp.mp4"
        output_path = self.config.output_dir / filename
        write_video(output_path, frames, fps=int(self.config.fps))
        self.state.last_warp_path = output_path
        return output_path, len(frames)

    def generate(
        self,
        *,
        prompt: str,
        image: Image.Image | None,
        active: set[str],
        translation_scale: float,
        rotation_degrees: float,
        chunks_per_generate: int,
        reset: bool,
        debug_warp: bool,
    ) -> dict[str, Any]:
        with self.lock:
            pipe = self.load_pipeline()
            if reset:
                self.reset()
            if self.state.wah_state is None:
                if image is None:
                    raise ValueError("Upload a first frame before the first generation.")
                if not prompt.strip():
                    raise ValueError("Prompt is required before the first generation.")
                self._init_generation(
                    pipe=pipe,
                    prompt=prompt.strip(),
                    image=image.convert("RGB"),
                    translation_scale=float(translation_scale),
                    chunks_per_generate=int(chunks_per_generate),
                )

            assert self.state.wah_state is not None
            assert self.state.current_pose is not None
            history_items = self.state.history_items
            if history_items is None:
                history_items = []
                self.state.history_items = history_items
            start_chunk = int(self.state.generated_chunks)
            previous_ui_scale = float(self.state.wah_state.get("web_ui_translation_scale", translation_scale))
            if self.state.generated_chunks > 0:
                multiplier = self.state.wah_state.get("web_ui_translation_effective_multiplier")
                if multiplier is None:
                    effective_scale = float(
                        self.state.wah_state.get("camera_translation_effective_scale", previous_ui_scale)
                    )
                    multiplier = effective_scale / previous_ui_scale if abs(previous_ui_scale) > 1.0e-8 else 1.0
                    self.state.wah_state["web_ui_translation_effective_multiplier"] = float(multiplier)
                self.state.wah_state["camera_translation_effective_scale"] = float(translation_scale) * float(multiplier)
            self.state.wah_state["camera_control_translation_scale"] = float(translation_scale)
            self.state.wah_state["web_ui_translation_scale"] = float(translation_scale)

            chunk_count = max(1, int(chunks_per_generate))
            for _ in range(chunk_count):
                is_first_generated_chunk = self.state.generated_chunks == 0
                camera_chunk, end_pose = build_camera_chunk(
                    self.state.current_pose,
                    active=active,
                    rotation_degrees=float(rotation_degrees),
                    window_num_frames=int(self.state.window_num_frames),
                    include_start=is_first_generated_chunk,
                )
                _, self.state.wah_state = pipe.generate_next_chunk(
                    self.state.wah_state,
                    camera_poses=camera_chunk,
                    output_type="latent",
                )
                if is_first_generated_chunk:
                    effective_scale = float(
                        self.state.wah_state.get("camera_translation_effective_scale", translation_scale)
                    )
                    multiplier = effective_scale / float(translation_scale) if abs(float(translation_scale)) > 1.0e-8 else 1.0
                    self.state.wah_state["web_ui_translation_effective_multiplier"] = float(multiplier)
                self.state.current_pose = end_pose
                self.state.generated_chunks += 1

            result = pipe.finalize_autoregressive_state(
                self.state.wah_state,
                output_type="np",
                free_model_hooks=False,
            )
            all_frames = unwrap_video_frames(result)
            previous_frame_count = sum(int(item.get("frames", 0)) for item in history_items)
            frames = all_frames[previous_frame_count:]
            if not frames:
                raise RuntimeError("No newly finalized frames were produced.")

            filename = f"{self.state.session_id}_{self.state.generated_chunks:04d}_new.mp4"
            output_path = self.config.output_dir / filename
            write_video(output_path, frames, fps=int(self.config.fps))
            self.state.last_video_path = output_path
            rel_name = output_path.name
            item = {
                "index": len(history_items),
                "label": f"Chunks {start_chunk + 1}-{self.state.generated_chunks}",
                "video_url": f"/media/{rel_name}?v={int(time.time() * 1000)}",
                "output_path": str(output_path),
                "frames": len(frames),
                "chunk_start": start_chunk + 1,
                "chunk_end": self.state.generated_chunks,
            }
            payload = {
                "ok": True,
                "video_url": item["video_url"],
                "output_path": str(output_path),
                "chunks_generated": self.state.generated_chunks,
                "window_num_frames": self.state.window_num_frames,
                "frames": len(frames),
                "active_controls": sorted(active),
                "history_item": item,
                "history": list(history_items) + [item],
            }
            if bool(debug_warp):
                warp_path, warp_frames = self._write_debug_warp_video(
                    frame_start=previous_frame_count,
                    frame_count=len(frames),
                )
                if warp_path is not None:
                    warp_name = warp_path.name
                    item["debug_warp_url"] = f"/media/{warp_name}?v={int(time.time() * 1000)}"
                    item["debug_warp_path"] = str(warp_path)
                    item["debug_warp_frames"] = int(warp_frames)
                    payload.update(
                        {
                            "debug_warp_url": item["debug_warp_url"],
                            "debug_warp_path": str(warp_path),
                            "debug_warp_frames": int(warp_frames),
                        }
                    )
                    payload["history_item"] = item
                    payload["history"] = list(history_items) + [item]
            history_items.append(item)
            return payload


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Warp-as-History Control</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #151719;
      --panel: #202429;
      --panel-2: #262b31;
      --line: #3a424b;
      --text: #eef2f5;
      --muted: #aab4bf;
      --accent: #6ec6a6;
      --accent-2: #f2b15d;
      --danger: #e06767;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.4 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .shell {
      display: grid;
      grid-template-columns: 320px minmax(480px, 1fr) 300px;
      gap: 14px;
      min-height: 100vh;
      padding: 14px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-width: 0;
    }
    .stage {
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 12px;
      min-width: 0;
    }
    .screen {
      width: 100%;
      min-height: 360px;
      background: #08090a;
      border: 1px solid var(--line);
      border-radius: 8px;
      display: grid;
      place-items: center;
      overflow: hidden;
    }
    .screen-grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 12px;
      min-width: 0;
    }
    .screen-grid.debug {
      grid-template-columns: 1fr 1fr;
      align-items: stretch;
    }
    .hidden { display: none; }
    video {
      width: 100%;
      height: 100%;
      max-height: calc(100vh - 160px);
      object-fit: contain;
      background: #08090a;
    }
    h1, h2 {
      margin: 0 0 10px;
      font-size: 15px;
      font-weight: 650;
      letter-spacing: 0;
    }
    label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin: 12px 0 6px;
    }
    textarea, input[type="number"], input[type="text"], input[type="file"] {
      width: 100%;
      background: var(--panel-2);
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 9px 10px;
      font: inherit;
    }
    textarea {
      min-height: 148px;
      resize: vertical;
    }
    input[type="range"] { width: 100%; }
    .grid3 {
      display: grid;
      grid-template-columns: repeat(3, 52px);
      grid-auto-rows: 44px;
      gap: 8px;
      justify-content: center;
      margin: 8px 0 14px;
    }
    .grid2 {
      display: grid;
      grid-template-columns: repeat(2, 64px);
      grid-auto-rows: 42px;
      gap: 8px;
      justify-content: center;
      margin: 8px 0 14px;
    }
    button {
      border: 1px solid var(--line);
      border-radius: 7px;
      color: var(--text);
      background: var(--panel-2);
      font: inherit;
      cursor: pointer;
    }
    .pad {
      font-size: 22px;
      line-height: 1;
    }
    .pad.active {
      border-color: var(--accent);
      background: color-mix(in srgb, var(--accent) 28%, var(--panel-2));
      color: #ffffff;
    }
    .primary {
      width: 100%;
      min-height: 46px;
      background: var(--accent);
      color: #07100c;
      border-color: transparent;
      font-weight: 700;
    }
    .secondary {
      width: 100%;
      min-height: 38px;
      margin-top: 10px;
    }
    .secondary.active {
      background: color-mix(in srgb, var(--accent-2) 30%, var(--panel-2));
      border-color: var(--accent-2);
      color: #fff7ed;
    }
    .danger { border-color: color-mix(in srgb, var(--danger) 70%, var(--line)); }
    .row {
      display: grid;
      grid-template-columns: 1fr 72px;
      gap: 10px;
      align-items: center;
      margin-bottom: 8px;
    }
    .status {
      min-height: 38px;
      color: var(--muted);
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 10px;
      overflow-wrap: anywhere;
    }
    .metric {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      padding: 7px 0;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
    }
    .metric strong { color: var(--text); font-weight: 600; }
    .history-list {
      display: grid;
      gap: 8px;
      max-height: 180px;
      overflow: auto;
      padding-top: 8px;
    }
    .history-item {
      width: 100%;
      min-height: 34px;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: center;
      padding: 7px 9px;
      text-align: left;
    }
    .history-item.active {
      border-color: var(--accent);
      background: color-mix(in srgb, var(--accent) 20%, var(--panel-2));
    }
    .history-item span:last-child {
      color: var(--muted);
      font-size: 12px;
    }
    @media (max-width: 1050px) {
      .shell { grid-template-columns: 1fr; }
      .screen-grid.debug { grid-template-columns: 1fr; }
      video { max-height: 62vh; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <section class="panel">
      <h1>Input</h1>
      <label for="firstFrame">First Frame</label>
      <input id="firstFrame" type="file" accept="image/*" />
      <label for="prompt">Prompt</label>
      <textarea id="prompt" spellcheck="false"></textarea>
      <button id="debugWarp" class="secondary" type="button">Debug Warp</button>
      <button id="generate" class="primary">Generate</button>
      <button id="reset" class="secondary danger">Reset State</button>
    </section>

    <section class="stage">
      <div class="status" id="status">Idle</div>
      <div class="screen-grid" id="screenGrid">
        <div class="screen">
          <video id="video" controls autoplay muted playsinline></video>
        </div>
        <div class="screen hidden" id="warpScreen">
          <video id="warpVideo" controls autoplay muted playsinline></video>
        </div>
      </div>
      <div class="panel">
        <div class="metric"><span>Output</span><strong id="outputPath">-</strong></div>
        <div class="metric"><span>Warp</span><strong id="warpPath">-</strong></div>
        <div class="metric"><span>Chunks</span><strong id="chunks">0</strong></div>
        <div class="metric"><span>Window</span><strong id="window">-</strong></div>
        <label>History</label>
        <div class="history-list" id="historyList"></div>
      </div>
    </section>

    <section class="panel">
      <h2>Move</h2>
      <div class="grid3">
        <span></span><button class="pad toggle" data-control="forward" title="Forward">W<br>↑</button><span></span>
        <button class="pad toggle" data-control="strafe_left" title="Strafe Left">A<br>←</button>
        <button class="pad toggle" data-control="backward" title="Backward">S<br>↓</button>
        <button class="pad toggle" data-control="strafe_right" title="Strafe Right">D<br>→</button>
        <button class="pad toggle" data-control="descend" title="Descend">Q</button>
        <span></span>
        <button class="pad toggle" data-control="rise" title="Rise">E</button>
      </div>

      <h2>Rotate</h2>
      <div class="grid3">
        <span></span><button class="pad toggle" data-control="pitch_up" title="Look Up">I<br>⇧</button><span></span>
        <button class="pad toggle" data-control="yaw_left" title="Turn Left">J<br>↶</button>
        <button class="pad toggle" data-control="pitch_down" title="Look Down">K<br>⇩</button>
        <button class="pad toggle" data-control="yaw_right" title="Turn Right">L<br>↷</button>
        <button class="pad toggle" data-control="roll_left" title="Roll Left">U<br>⟲</button>
        <span></span>
        <button class="pad toggle" data-control="roll_right" title="Roll Right">O<br>⟳</button>
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
      <label>Chunks Per Generate</label>
      <div class="row">
        <input id="chunksPerGenerate" type="range" min="1" max="4" step="1" value="1" />
        <input id="chunksPerGenerateValue" type="number" min="1" max="16" step="1" value="1" />
      </div>
    </section>
  </main>
  <script>
    const active = new Set();
    const keyMap = {
      KeyW: 'forward',
      ArrowUp: 'forward',
      KeyS: 'backward',
      ArrowDown: 'backward',
      KeyA: 'strafe_left',
      KeyD: 'strafe_right',
      KeyQ: 'descend',
      KeyE: 'rise',
      KeyJ: 'yaw_left',
      ArrowLeft: 'yaw_left',
      KeyL: 'yaw_right',
      ArrowRight: 'yaw_right',
      KeyI: 'pitch_up',
      KeyK: 'pitch_down',
      KeyU: 'roll_left',
      KeyO: 'roll_right'
    };
    const statusEl = document.getElementById('status');
    const videoEl = document.getElementById('video');
    const warpVideoEl = document.getElementById('warpVideo');
    const warpScreenEl = document.getElementById('warpScreen');
    const screenGridEl = document.getElementById('screenGrid');
    const outputEl = document.getElementById('outputPath');
    const warpPathEl = document.getElementById('warpPath');
    const chunksEl = document.getElementById('chunks');
    const windowEl = document.getElementById('window');
    const historyListEl = document.getElementById('historyList');
    const debugWarpButton = document.getElementById('debugWarp');
    let debugWarpEnabled = false;
    let forceReset = false;
    let historyItems = [];
    let selectedHistoryIndex = -1;

    for (const button of document.querySelectorAll('.toggle')) {
      button.addEventListener('click', () => {
        const key = button.dataset.control;
        setControl(key, !active.has(key));
      });
    }

    function setControl(key, enabled) {
      if (enabled) active.add(key);
      else active.delete(key);
      for (const twin of document.querySelectorAll(`[data-control="${key}"]`)) {
        twin.classList.toggle('active', active.has(key));
      }
    }

    window.addEventListener('keydown', (event) => {
      if (event.target && ['INPUT', 'TEXTAREA'].includes(event.target.tagName)) return;
      const key = keyMap[event.code];
      if (!key) return;
      event.preventDefault();
      setControl(key, true);
    });

    window.addEventListener('keyup', (event) => {
      if (event.target && ['INPUT', 'TEXTAREA'].includes(event.target.tagName)) return;
      const key = keyMap[event.code];
      if (!key) return;
      event.preventDefault();
      setControl(key, false);
    });

    function bindPair(sliderId, inputId) {
      const slider = document.getElementById(sliderId);
      const input = document.getElementById(inputId);
      slider.addEventListener('input', () => { input.value = slider.value; });
      input.addEventListener('input', () => { slider.value = input.value; });
    }
    bindPair('translationScale', 'translationScaleValue');
    bindPair('rotationDegrees', 'rotationDegreesValue');
    bindPair('chunksPerGenerate', 'chunksPerGenerateValue');

    debugWarpButton.addEventListener('click', () => {
      debugWarpEnabled = !debugWarpEnabled;
      debugWarpButton.classList.toggle('active', debugWarpEnabled);
      if (!debugWarpEnabled) clearWarpVideo();
    });

    function clearWarpVideo() {
      warpPathEl.textContent = '-';
      warpVideoEl.removeAttribute('src');
      warpVideoEl.load();
      warpScreenEl.classList.add('hidden');
      screenGridEl.classList.remove('debug');
    }

    function showHistoryItem(item) {
      if (!item) return;
      selectedHistoryIndex = item.index;
      videoEl.src = item.video_url;
      videoEl.load();
      outputEl.textContent = item.output_path || '-';
      if (debugWarpEnabled && item.debug_warp_url) {
        warpVideoEl.src = item.debug_warp_url;
        warpVideoEl.load();
        warpPathEl.textContent = item.debug_warp_path || '-';
        warpScreenEl.classList.remove('hidden');
        screenGridEl.classList.add('debug');
      } else {
        clearWarpVideo();
      }
      renderHistory();
    }

    function renderHistory() {
      historyListEl.innerHTML = '';
      for (const item of historyItems) {
        const button = document.createElement('button');
        button.className = 'history-item';
        button.classList.toggle('active', item.index === selectedHistoryIndex);
        button.type = 'button';
        const label = document.createElement('span');
        label.textContent = item.label || `Chunk ${item.index + 1}`;
        const meta = document.createElement('span');
        meta.textContent = `${item.frames || 0}f`;
        button.append(label, meta);
        button.addEventListener('click', () => showHistoryItem(item));
        historyListEl.append(button);
      }
    }

    document.getElementById('reset').addEventListener('click', async () => {
      forceReset = true;
      const res = await fetch('/api/reset', { method: 'POST' });
      const payload = await res.json();
      statusEl.textContent = payload.message || 'Reset';
      chunksEl.textContent = '0';
      windowEl.textContent = '-';
      outputEl.textContent = '-';
      videoEl.removeAttribute('src');
      videoEl.load();
      clearWarpVideo();
      historyItems = [];
      selectedHistoryIndex = -1;
      renderHistory();
    });

    document.getElementById('generate').addEventListener('click', async () => {
      const generateButton = document.getElementById('generate');
      generateButton.disabled = true;
      statusEl.textContent = 'Generating...';
      try {
        const form = new FormData();
        const fileInput = document.getElementById('firstFrame');
        if (fileInput.files.length) form.append('first_frame', fileInput.files[0]);
        form.append('prompt', document.getElementById('prompt').value);
        form.append('controls', JSON.stringify({ active: Array.from(active) }));
        form.append('translation_scale', document.getElementById('translationScaleValue').value);
        form.append('rotation_degrees', document.getElementById('rotationDegreesValue').value);
        form.append('chunks_per_generate', document.getElementById('chunksPerGenerateValue').value);
        form.append('debug_warp', debugWarpEnabled ? '1' : '0');
        form.append('reset', forceReset ? '1' : '0');
        const res = await fetch('/api/generate', { method: 'POST', body: form });
        const payload = await res.json();
        if (!res.ok || !payload.ok) throw new Error(payload.error || 'Generation failed');
        forceReset = false;
        historyItems = payload.history || (payload.history_item ? [...historyItems, payload.history_item] : historyItems);
        showHistoryItem(payload.history_item || historyItems[historyItems.length - 1]);
        chunksEl.textContent = String(payload.chunks_generated);
        windowEl.textContent = String(payload.window_num_frames);
        statusEl.textContent = `Done: ${payload.frames} frames`;
      } catch (error) {
        statusEl.textContent = error.message;
      } finally {
        generateButton.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


def make_handler(app: WarpControlApp):
    class Handler(BaseHTTPRequestHandler):
        server_version = "WarpControlHTTP/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                _text_response(self, HTML_PAGE)
                return
            if parsed.path.startswith("/media/"):
                self._serve_media(parsed.path)
                return
            _text_response(self, "Not found", status=HTTPStatus.NOT_FOUND, content_type="text/plain; charset=utf-8")

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/api/reset":
                with app.lock:
                    app.reset()
                _json_response(self, {"ok": True, "message": "State reset"})
                return
            if parsed.path == "/api/generate":
                self._handle_generate()
                return
            _json_response(self, {"ok": False, "error": "Not found"}, status=HTTPStatus.NOT_FOUND)

        def _serve_media(self, path: str) -> None:
            rel = unquote(path[len("/media/") :]).split("?", 1)[0]
            if not rel or "/" in rel or "\\" in rel:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            media_path = (app.config.output_dir / rel).resolve()
            if not media_path.is_file() or app.config.output_dir.resolve() not in media_path.parents:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            data = media_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _handle_generate(self) -> None:
            try:
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                        "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
                    },
                )
                image = None
                image_item = form["first_frame"] if "first_frame" in form else None
                if image_item is not None and getattr(image_item, "filename", ""):
                    image_bytes = image_item.file.read()
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

                controls_raw = form.getfirst("controls", "{}")
                controls = json.loads(controls_raw)
                active = {str(item) for item in controls.get("active", [])}
                payload = app.generate(
                    prompt=str(form.getfirst("prompt", "")),
                    image=image,
                    active=active,
                    translation_scale=float(form.getfirst("translation_scale", "0.1")),
                    rotation_degrees=float(form.getfirst("rotation_degrees", "4")),
                    chunks_per_generate=int(float(form.getfirst("chunks_per_generate", "1"))),
                    reset=str(form.getfirst("reset", "0")) == "1",
                    debug_warp=str(form.getfirst("debug_warp", "0")) == "1",
                )
                _json_response(self, payload)
            except Exception as exc:
                _json_response(
                    self,
                    {"ok": False, "error": html.escape(str(exc))},
                    status=HTTPStatus.BAD_REQUEST,
                )

    return Handler


def main() -> None:
    args = parse_args()
    if not args.enable_optional_attention:
        disable_diffusers_optional_attention()
    config = AppConfig(
        model_path=resolve_model_path(args.model_path),
        lora_path=args.lora_path or None,
        height=int(args.height),
        width=int(args.width),
        fps=int(args.fps),
        seed=int(args.seed),
        device=str(args.device),
        dtype=str(args.dtype),
        output_dir=args.output_dir.expanduser().resolve(),
    )
    app = WarpControlApp(config)
    if args.preload:
        app.load_pipeline()
    server = ThreadingHTTPServer((str(args.host), int(args.port)), make_handler(app))
    print(
        json.dumps(
            {
                "event": "web_control_ready",
                "url": f"http://{args.host}:{args.port}",
                "output_dir": str(config.output_dir),
                "model_path": config.model_path,
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
