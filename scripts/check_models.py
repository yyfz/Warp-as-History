#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


HELIOS_DIR = REPO_ROOT / "checkpoints" / "helios-distilled"
HELIOS_MID_DIR = REPO_ROOT / "checkpoints" / "helios-mid"
PI3_REPO = REPO_ROOT / "third_party" / "Pi3"
PI3X_CKPT = REPO_ROOT / "checkpoints" / "pi3x" / "model.safetensors"
HELIOS_DOWNLOAD_COMMAND = (
    "huggingface-cli download BestWishYsh/Helios-Distilled "
    "--local-dir checkpoints/helios-distilled"
)
HELIOS_MID_DOWNLOAD_COMMAND = (
    "huggingface-cli download BestWishYsh/Helios-Mid "
    "--local-dir checkpoints/helios-mid"
)
PI3X_DOWNLOAD_COMMAND = (
    "huggingface-cli download yyfz233/Pi3X model.safetensors "
    "--local-dir checkpoints/pi3x"
)


def format_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def check_pi3_repo(errors: list[str]) -> None:
    marker = PI3_REPO / "pi3" / "models" / "pi3x.py"
    if marker.is_file():
        print(f"[ok] Pi3 submodule: {PI3_REPO}")
        return
    errors.append(
        "Missing Pi3 submodule. Run `git submodule update --init --recursive` "
        f"and check that {marker} exists."
    )


def check_helios_model(errors: list[str]) -> None:
    required = [
        HELIOS_DIR / "model_index.json",
        HELIOS_DIR / "scheduler" / "scheduler_config.json",
        HELIOS_DIR / "transformer" / "config.json",
        HELIOS_DIR / "vae" / "config.json",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        missing_text = "\n  ".join(str(path) for path in missing)
        errors.append(f"Missing Helios-Distilled files:\n  {missing_text}\nDownload it with:\n  {HELIOS_DOWNLOAD_COMMAND}")
        return
    print(f"[ok] Helios-Distilled: {HELIOS_DIR}")


def check_helios_mid(warnings: list[str]) -> None:
    required = [
        HELIOS_MID_DIR / "model_index.json",
        HELIOS_MID_DIR / "transformer" / "config.json",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        missing_text = "\n  ".join(str(path) for path in missing)
        warnings.append(
            "Helios-Mid is not present. Inference does not need it, but download it before "
            f"training if you use the Mid transformer:\n  {missing_text}\nDownload it with:\n  "
            f"{HELIOS_MID_DOWNLOAD_COMMAND}"
        )
        return
    print(f"[ok] Helios-Mid: {HELIOS_MID_DIR}")


def check_pi3x_checkpoint(errors: list[str]) -> None:
    ckpt = PI3X_CKPT
    if not ckpt.is_file():
        errors.append(f"Missing Pi3X checkpoint: {ckpt}\nDownload it with:\n  {PI3X_DOWNLOAD_COMMAND}")
        return

    try:
        from safetensors import safe_open
    except Exception as exc:
        errors.append(f"Could not import safetensors to inspect {ckpt}: {exc}")
        return

    try:
        with safe_open(str(ckpt), framework="pt", device="cpu") as handle:
            tensor_count = len(handle.keys())
    except Exception as exc:
        errors.append(f"Pi3X checkpoint is not a readable safetensors file: {ckpt}\n{exc}")
        return

    size = format_size(ckpt.stat().st_size)
    print(f"[ok] Pi3X checkpoint: {ckpt} ({size}, {tensor_count} tensors)")


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    check_helios_model(errors)
    check_helios_mid(warnings)
    check_pi3_repo(errors)
    check_pi3x_checkpoint(errors)
    if warnings:
        print()
        for warning in warnings:
            print(f"[warn] {warning}")
    if errors:
        print()
        for error in errors:
            print(f"[error] {error}", file=sys.stderr)
        return 1
    print("[ok] Model prerequisites are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
