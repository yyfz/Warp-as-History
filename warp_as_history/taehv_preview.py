from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


TAEHV_VAE_MODES = frozenset({"off", "decode", "full"})


class _DotConfig(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class _DeterministicLatentDistribution:
    def __init__(self, latents: torch.Tensor):
        self.latents = latents

    def sample(self, generator: torch.Generator | list[torch.Generator] | None = None) -> torch.Tensor:
        return self.latents


class _EncoderOutput:
    def __init__(self, latents: torch.Tensor):
        self.latent_dist = _DeterministicLatentDistribution(latents)


class TAEHVDiffusersWrapper(nn.Module):
    """Diffusers-shaped wrapper around TAEHV.

    TAEHV works on NTCHW videos in [0, 1] and consumes diffusion latents directly.
    The surrounding Helios/Warp-as-History pipeline uses NCTHW videos in [-1, 1],
    so this wrapper only handles layout/range adaptation.
    """

    def __init__(self, taehv: nn.Module, *, parallel: bool = True, reference_config: Any | None = None):
        super().__init__()
        self.taehv = taehv
        self.parallel = bool(parallel)
        latent_channels = int(getattr(taehv, "latent_channels", 16))
        scale_factor_temporal = int(getattr(reference_config, "scale_factor_temporal", getattr(taehv, "t_upscale", 4)))
        scale_factor_spatial = int(
            getattr(reference_config, "scale_factor_spatial", 8 * int(getattr(taehv, "patch_size", 1)))
        )
        self.config = _DotConfig(
            z_dim=latent_channels,
            latents_mean=[0.0] * latent_channels,
            latents_std=[1.0] * latent_channels,
            scale_factor_temporal=scale_factor_temporal,
            scale_factor_spatial=scale_factor_spatial,
        )

    @property
    def dtype(self) -> torch.dtype:
        return next(self.taehv.parameters()).dtype

    def encode(self, video: torch.Tensor) -> _EncoderOutput:
        if video.ndim != 5:
            raise ValueError(f"Expected NCTHW video tensor, got {tuple(video.shape)}")
        video_ntchw = video.to(dtype=self.dtype).permute(0, 2, 1, 3, 4).div(2.0).add(0.5).clamp_(0.0, 1.0)
        pad = int(getattr(self.taehv, "frames_to_trim", 3))
        if pad > 0:
            video_ntchw = torch.cat([video_ntchw[:, :1].expand(-1, pad, -1, -1, -1), video_ntchw], dim=1)
        latents_ntchw = self.taehv.encode_video(
            video_ntchw,
            parallel=self.parallel,
        )
        latents = latents_ntchw.permute(0, 2, 1, 3, 4)
        return _EncoderOutput(latents)

    def decode_diffusion_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 5:
            raise ValueError(f"Expected NCTHW latent tensor, got {tuple(latents.shape)}")
        latents_ntchw = latents.to(dtype=self.dtype).permute(0, 2, 1, 3, 4)
        video_ntchw = self.taehv.decode_video(
            latents_ntchw,
            parallel=self.parallel,
        )
        trim = int(getattr(self.taehv, "frames_to_trim", 3))
        if trim > 0 and video_ntchw.shape[1] > trim:
            video_ntchw = video_ntchw[:, trim:]
        return video_ntchw.permute(0, 2, 1, 3, 4).mul(2.0).sub(1.0)

    def decode(self, latents: torch.Tensor, return_dict: bool | None = None) -> tuple[torch.Tensor]:
        return (self.decode_diffusion_latents(latents),)


@dataclass
class TAEHVPreviewBackend:
    mode: str
    checkpoint: Path
    vae: TAEHVDiffusersWrapper

    @property
    def decoder(self) -> TAEHVDiffusersWrapper:
        return self.vae


def _taehv_import_error_message(exc: Exception) -> str:
    return (
        "TAEHV VAE was requested, but Python module 'taehv' is not importable. "
        "Use the vendored third_party/taehv module or put taehv.py on PYTHONPATH, then retry. "
        "Repository: https://github.com/madebyollin/taehv. "
        f"Original import error: {type(exc).__name__}: {exc}"
    )


def validate_taehv_checkpoint(mode: str, checkpoint: str | Path | None) -> Path | None:
    mode = str(mode or "off").strip().lower()
    if mode not in TAEHV_VAE_MODES:
        raise ValueError("taehv_vae_mode must be one of: off, decode, full.")
    if mode == "off":
        return None
    if checkpoint is None or not str(checkpoint).strip():
        raise FileNotFoundError(
            "TAEHV VAE mode is enabled but --taehv_checkpoint was not provided. "
            "Download taew2_1.pth from https://github.com/madebyollin/taehv and pass "
            "--taehv_checkpoint /path/to/taew2_1.pth."
        )
    checkpoint_path = Path(checkpoint).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing TAEHV checkpoint: {checkpoint_path}. Download it with:\n"
            "  mkdir -p checkpoints/taehv\n"
            "  wget -O checkpoints/taehv/taew2_1.pth "
            "https://github.com/madebyollin/taehv/raw/main/taew2_1.pth\n"
            "Then pass --taehv_checkpoint checkpoints/taehv/taew2_1.pth."
        )
    return checkpoint_path


def create_taehv_preview_backend(
    *,
    mode: str,
    checkpoint: str | Path | None,
    device: torch.device | str,
    dtype: torch.dtype,
    reference_config: Any | None = None,
    parallel: bool = True,
) -> TAEHVPreviewBackend | None:
    mode = str(mode or "off").strip().lower()
    if mode not in TAEHV_VAE_MODES:
        raise ValueError("taehv_vae_mode must be one of: off, decode, full.")
    if mode == "off":
        return None
    checkpoint_path = validate_taehv_checkpoint(mode, checkpoint)
    assert checkpoint_path is not None

    try:
        from taehv import TAEHV
    except Exception as exc:  # pragma: no cover - depends on optional user install
        raise ImportError(_taehv_import_error_message(exc)) from exc

    taehv = TAEHV(checkpoint_path=str(checkpoint_path))
    taehv.eval().requires_grad_(False).to(device=device, dtype=dtype)
    return TAEHVPreviewBackend(
        mode=mode,
        checkpoint=checkpoint_path.resolve(),
        vae=TAEHVDiffusersWrapper(taehv, parallel=parallel, reference_config=reference_config),
    )
