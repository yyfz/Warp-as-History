from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from diffusers import AutoencoderKLWan
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3

from helios.diffusers_version.pipeline_helios_diffusers import HeliosPipeline, calculate_shift
from helios.diffusers_version.scheduling_helios_diffusers import HeliosScheduler
from helios.modules.transformer_helios import HeliosTransformer3DModel as TrainingHeliosTransformer3DModel
from helios.utils.utils_base import apply_schedule_shift
from helios.modules.helios_kernels import (
    replace_all_norms_with_flash_norms,
    replace_rmsnorm_with_fp32,
    replace_rope_with_flash_rope,
)

NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
    "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, "
    "messy background, three legs, many people in the background, walking backwards"
)

DEFAULT_TRAINING_ARGS = {
    "add_noise_to_video_latents": True,
    "attention_backend": "auto",
    "base_model_path": "checkpoints/helios-distilled",
    "data_root": "data/training",
    "flow_matching_mode": "train_exact",
    "flow_matching_stage_id": 0,
    "flow_matching_stage_sampling": "all",
    "flow_matching_train_exact_timestep_sampling": "training_density",
    "flow_matching_use_dynamic_shifting": "off",
    "gradient_checkpointing": False,
    "height": 384,
    "history_assignment": "contiguous",
    "history_assignment_seed": 0,
    "history_invisible_token_mode": "none",
    "history_invisible_token_threshold": 0.1,
    "history_perturbation": "none",
    "history_sizes": [16, 2, 1],
    "history_temporal_layout": "long_mid_short",
    "history_visible_token_drop": False,
    "history_visible_token_threshold": 0.1,
    "image_noise_sigma_max": 0.135,
    "image_noise_sigma_min": 0.111,
    "is_amplify_first_chunk": False,
    "iters": 300,
    "limit": None,
    "log_every": 1,
    "lora_adapter_name": "visible_fit",
    "lora_alpha": 1,
    "lora_dropout": 0.0,
    "lora_rank": 1,
    "lora_target_modules": "to_q,to_k,to_v",
    "lr": 0.03,
    "lr_schedule": "constant",
    "lr_schedule_final_ratio": 1.0,
    "max_grad_norm": 1.0,
    "max_steps": 1500,
    "num_frames": 33,
    "num_latent_frames_per_chunk": 9,
    "output_dir": "runs/warp_as_history_lora",
    "overwrite": False,
    "prompt_cache_dir": "",
    "prompt_csv": "data/training/training_data.csv",
    "pyramid_num_inference_steps_list": [1, 1, 1],
    "history_positioning": "none",
    "history_position_count": 9,
    "history_position_delta": 1,
    "save_every": 500,
    "seed": 42,
    "shuffle": True,
    "transformer_path": "checkpoints/helios-distilled",
    "use_warp_as_history": False,
    "video_noise_sigma_max": 0.135,
    "video_noise_sigma_min": 0.111,
    "weighting_scheme": "none",
    "width": 640,
}


def parse_args(argv=None):
    return argparse.Namespace(**DEFAULT_TRAINING_ARGS.copy())


def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def stable_seed_from_parts(base_seed: int, *parts) -> int:
    payload = "|".join([str(int(base_seed))] + [str(part) for part in parts]).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**63 - 1)

def seed_global_rng(seed: int):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    np.random.seed(int(seed) % (2**32))

def resolved_attention_backend(args) -> str:
    if args.attention_backend != "auto":
        return args.attention_backend
    return "flash"

def load_pipeline(args, device):
    attention_backend = resolved_attention_backend(args)
    print(json.dumps({"event": "load_pipeline_transformer_impl", "transformer_impl": "training_modules"}), flush=True)
    transformer = TrainingHeliosTransformer3DModel.from_pretrained(
        args.transformer_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    transformer = replace_rmsnorm_with_fp32(transformer)
    transformer = replace_all_norms_with_flash_norms(transformer)
    replace_rope_with_flash_rope()
    transformer.set_attention_backend(attention_backend)
    print(json.dumps({"event": "load_pipeline_backend", "attention_backend": attention_backend}), flush=True)
    if args.gradient_checkpointing and hasattr(transformer, "enable_gradient_checkpointing"):
        transformer.enable_gradient_checkpointing()

    vae = AutoencoderKLWan.from_pretrained(args.base_model_path, subfolder="vae", torch_dtype=torch.float32)
    scheduler = HeliosScheduler.from_pretrained(args.base_model_path, subfolder="scheduler")
    pipe = HeliosPipeline.from_pretrained(
        args.base_model_path,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        torch_dtype=torch.bfloat16,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    for module in (pipe.text_encoder, pipe.transformer, pipe.vae):
        module.eval()
        module.requires_grad_(False)
    return pipe

def latent_stats(pipe, device):
    mean = torch.tensor(pipe.vae.config.latents_mean).view(1, pipe.vae.config.z_dim, 1, 1, 1)
    std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1)
    return mean.to(device, pipe.vae.dtype), std.to(device, pipe.vae.dtype)

def encode_video_latents(pipe, frames, args, device, mean, std):
    video = pipe.video_processor.preprocess_video(frames, height=args.height, width=args.width)
    video = video.to(device=device, dtype=pipe.vae.dtype)
    with torch.no_grad():
        dist = pipe.vae.encode(video).latent_dist
        latents = dist.mode() if hasattr(dist, "mode") else dist.mean
        latents = (latents - mean) * std
    return latents.float()

def prepare_condition(
    pipe,
    first_frame,
    prompt,
    args,
    device,
    mean,
    std,
    history_frames=None,
    prompt_embeds_override=None,
):
    generator = torch.Generator(device=device).manual_seed(args.seed)
    with torch.no_grad():
        if prompt_embeds_override is None:
            prompt_embeds, _negative_prompt_embeds = pipe.encode_prompt(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                do_classifier_free_guidance=False,
                num_videos_per_prompt=1,
                max_sequence_length=512,
                device=device,
            )
        else:
            prompt_embeds = prompt_embeds_override
        video_latents = None
        if history_frames is None:
            image = pipe.video_processor.preprocess(first_frame, height=args.height, width=args.width)
            image_latents, fake_image_latents = pipe.prepare_image_latents(
                image,
                latents_mean=mean,
                latents_std=std,
                num_latent_frames_per_chunk=args.num_latent_frames_per_chunk,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
        else:
            condition_image_latents = None
            condition_fake_image_latents = None
            if getattr(args, "use_warp_as_history", False):
                image = pipe.video_processor.preprocess(first_frame, height=args.height, width=args.width)
                condition_image_latents, condition_fake_image_latents = pipe.prepare_image_latents(
                    image,
                    latents_mean=mean,
                    latents_std=std,
                    num_latent_frames_per_chunk=args.num_latent_frames_per_chunk,
                    dtype=torch.float32,
                    device=device,
                    generator=generator,
                )
            video = pipe.video_processor.preprocess_video(history_frames, height=args.height, width=args.width)
            image_latents, video_latents = pipe.prepare_video_latents(
                video,
                latents_mean=mean,
                latents_std=std,
                num_latent_frames_per_chunk=args.num_latent_frames_per_chunk,
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            if getattr(args, "use_warp_as_history", False):
                image_latents = condition_image_latents.to(device=device, dtype=torch.float32)
                fake_image_latents = condition_fake_image_latents.to(device=device, dtype=torch.float32)
            else:
                fake_image_latents = None
    transformer_dtype = transformer_compute_dtype(pipe.transformer)
    prompt_embeds = prompt_embeds.to(transformer_dtype)
    return (
        prompt_embeds,
        image_latents.float(),
        fake_image_latents.float() if fake_image_latents is not None else None,
        video_latents.float() if video_latents is not None else None,
    )

def apply_history_video_latent_perturbation(video_latents, args, seq=None):
    mode = getattr(args, "history_perturbation", "none")
    if video_latents is None or mode in {"none", "wrong_sequence"}:
        return video_latents
    if mode == "zero_video_latents":
        perturbed = torch.zeros_like(video_latents)
    elif mode == "shuffle_video_latents":
        frames = video_latents.shape[2]
        generator = torch.Generator(device="cpu").manual_seed(
            stable_seed_from_parts(int(args.seed), "history_shuffle", seq or "unknown")
        )
        order = torch.randperm(frames, generator=generator).to(device=video_latents.device)
        perturbed = video_latents.index_select(2, order)
    else:
        raise ValueError(f"Unsupported --history_perturbation: {mode}")
    print(
        json.dumps(
            {
                "event": "history_perturbation",
                "seq": seq,
                "mode": mode,
                "video_history_latent_frames": int(video_latents.shape[2]),
            }
        ),
        flush=True,
    )
    return perturbed


def add_official_history_latent_noise(latents, args, *, sigma_prefix, per_frame, seq=None, event):
    if latents is None or not bool(getattr(args, "add_noise_to_video_latents", True)):
        return latents
    sigma_min = float(getattr(args, f"{sigma_prefix}_noise_sigma_min", 0.111))
    sigma_max = float(getattr(args, f"{sigma_prefix}_noise_sigma_max", 0.135))
    if sigma_min < 0.0 or sigma_max < sigma_min:
        raise ValueError(
            f"{sigma_prefix} noise sigma range must satisfy 0 <= min <= max, got [{sigma_min}, {sigma_max}]."
        )
    if per_frame:
        frames = int(latents.shape[2])
        sigmas = torch.rand(frames, device=latents.device, dtype=torch.float32) * (sigma_max - sigma_min) + sigma_min
        sigmas = sigmas.view(1, 1, frames, 1, 1).to(dtype=latents.dtype)
    else:
        sigmas = (
            torch.rand(1, device=latents.device, dtype=torch.float32) * (sigma_max - sigma_min) + sigma_min
        ).to(dtype=latents.dtype)
        sigmas = sigmas.view(1, 1, 1, 1, 1)
    noisy = sigmas * torch.randn(latents.shape, device=latents.device, dtype=latents.dtype) + (1 - sigmas) * latents
    print(
        json.dumps(
            {
                "event": event,
                "seq": seq,
                "sigma_min": sigma_min,
                "sigma_max": sigma_max,
                "per_frame": bool(per_frame),
                "latent_frames": int(latents.shape[2]),
            }
        ),
        flush=True,
    )
    return noisy


def add_official_video_history_noise(video_latents, args, seq=None):
    return add_official_history_latent_noise(
        video_latents,
        args,
        sigma_prefix="video",
        per_frame=True,
        seq=seq,
        event="video_history_latent_noise",
    )


def add_official_image_history_prefix_noise(image_latents, args, seq=None):
    return add_official_history_latent_noise(
        image_latents,
        args,
        sigma_prefix="image",
        per_frame=False,
        seq=seq,
        event="image_history_prefix_noise",
    )


def remap_history_rope_indices(
    indices_latents_history_long,
    indices_latents_history_mid,
    indices_latents_history_1x,
    indices_hidden_states,
    args,
    video_count,
    history_offset,
    seq=None,
):
    mode = getattr(args, "history_positioning", "none")
    if mode == "none":
        return indices_latents_history_long, indices_latents_history_mid, indices_latents_history_1x
    if mode not in {"last_n", "last_n_same_order"}:
        raise ValueError(f"Unsupported --history_positioning: {mode}")
    if video_count <= 0:
        print(
            json.dumps(
                {
                    "event": "history_positioning_skipped",
                    "seq": seq,
                    "reason": "no_video_history_latents",
                    "mode": mode,
                }
            ),
            flush=True,
        )
        return indices_latents_history_long, indices_latents_history_mid, indices_latents_history_1x

    history_indices = torch.cat(
        [indices_latents_history_long, indices_latents_history_mid, indices_latents_history_1x], dim=0
    ).clone()
    count = min(
        int(getattr(args, "history_position_count", args.num_latent_frames_per_chunk)),
        int(video_count),
        int(args.num_latent_frames_per_chunk),
    )
    if count <= 0:
        return indices_latents_history_long, indices_latents_history_mid, indices_latents_history_1x
    delta = int(getattr(args, "history_position_delta", 1))
    selected_start = int(history_offset + video_count - count)
    selected_end = selected_start + count
    target_start = int(indices_hidden_states[0].item())
    if mode == "last_n":
        remapped = target_start + torch.arange(count - 1, -1, -1, device=history_indices.device) - delta
    else:
        remapped = target_start + torch.arange(count, device=history_indices.device) - delta
    if int(remapped.min().item()) < 0:
        raise ValueError(f"--history_position_delta={delta} makes negative history indices: {remapped.tolist()}")

    old_indices = history_indices[selected_start:selected_end].detach().cpu().tolist()
    history_indices[selected_start:selected_end] = remapped.to(history_indices)
    new_indices = history_indices[selected_start:selected_end].detach().cpu().tolist()
    history_sizes = [int(x) for x in args.history_sizes]
    remapped_long, remapped_mid, remapped_1x = history_indices.split(history_sizes, dim=0)
    print(
        json.dumps(
            {
                "event": "history_positioning",
                "seq": seq,
                "mode": mode,
                "count": int(count),
                "delta": int(delta),
                "video_count": int(video_count),
                "history_offset": int(history_offset),
                "selected_history_positions": [int(selected_start), int(selected_end)],
                "target_start_index": int(target_start),
                "old_indices": old_indices,
                "new_indices": new_indices,
            }
        ),
        flush=True,
    )
    return remapped_long, remapped_mid, remapped_1x

def make_histories(pipe, image_latents, fake_image_latents, args, device, video_latents=None, seq=None):
    batch_size = 1
    channels = pipe.transformer.config.in_channels
    history_sizes = [int(x) for x in args.history_sizes]
    history_dtype = transformer_compute_dtype(pipe.transformer)
    h = args.height // pipe.vae_scale_factor_spatial
    w = args.width // pipe.vae_scale_factor_spatial
    num_history_latents = sum(history_sizes)
    history_latents = torch.zeros(batch_size, channels, num_history_latents, h, w, device=device, dtype=torch.float32)
    history_visible_latents = None
    video_count = 0
    history_offset = num_history_latents

    def _pad_mask_for_3d_conv(x, kernel_size):
        _b, _c, t, hh, ww = x.shape
        pt, ph, pw = kernel_size
        pad_t = (pt - (t % pt)) % pt
        pad_h = (ph - (hh % ph)) % ph
        pad_w = (pw - (ww % pw)) % pw
        return F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode="replicate")

    def _estimate_patch_keep(mask_tensor, patch_size, threshold):
        if mask_tensor is None:
            return None
        pooled = _pad_mask_for_3d_conv(mask_tensor.float(), patch_size)
        pooled = F.avg_pool3d(pooled, kernel_size=patch_size, stride=patch_size)
        return pooled >= float(threshold)

    def _load_video_visibility_latents(video_frames_count):
        if video_frames_count <= 0:
            return None
        if args.use_warp_as_history:
            extra_mask_frames = getattr(args, "history_visibility_extra_mask_frames", None)
            if not extra_mask_frames:
                raise ValueError(
                    "Warp history visibility requires history_visibility_extra_mask_frames from the training renderer."
                )
            if len(extra_mask_frames) < int(args.num_frames):
                raise ValueError(
                    "history_visibility_extra_mask_frames has "
                    f"{len(extra_mask_frames)} frames, need at least {int(args.num_frames)}."
                )
            mask = np.stack(
                [
                    np.asarray(mask_frame.convert("L"), dtype=np.float32) / 255.0
                    for mask_frame in extra_mask_frames[: int(args.num_frames)]
                ],
                axis=0,
            )
            if mask.shape[0] < int(args.num_frames):
                raise ValueError(
                    f"History visibility mask produced {mask.shape[0]} frames, need {int(args.num_frames)}."
                )
            sampled_ids = np.arange(video_frames_count, dtype=np.int64) * int(pipe.vae_scale_factor_temporal)
            sampled_ids = np.clip(sampled_ids, 0, mask.shape[0] - 1)
            sampled = torch.from_numpy(mask[sampled_ids]).to(device=device, dtype=torch.float32)
            sampled = sampled.unsqueeze(0).unsqueeze(0)
            sampled = F.interpolate(sampled, size=(video_frames_count, h, w), mode="trilinear", align_corners=False)
            sampled = sampled.clamp_(0.0, 1.0)
            return sampled
        return None

    if bool(getattr(args, "use_warp_as_history", False)):
        if video_latents is None:
            raise ValueError("--use_warp_as_history requires warp video latents.")
        video_latents = apply_history_video_latent_perturbation(video_latents, args, seq=seq)
        video_latents = add_official_video_history_noise(video_latents, args, seq=seq)

        latent_window_size = int(args.num_latent_frames_per_chunk)
        video_frames = int(video_latents.shape[2])
        warp_latents = torch.zeros(
            batch_size,
            channels,
            latent_window_size,
            h,
            w,
            device=device,
            dtype=torch.float32,
        )
        warp_count = min(video_frames, latent_window_size)
        if warp_count > 0:
            warp_latents[:, :, -warp_count:] = video_latents[:, :, -warp_count:]

        visible_history_short = None
        visible_history_mid = None
        visible_history_long = None
        if getattr(args, "history_visible_token_drop", False) or using_history_invisible_token(args):
            video_visible_latents = _load_video_visibility_latents(video_frames)
            if video_visible_latents is not None:
                visible_history_short = torch.zeros(
                    batch_size,
                    1,
                    latent_window_size,
                    h,
                    w,
                    device=device,
                    dtype=torch.float32,
                )
                if warp_count > 0:
                    visible_history_short[:, :, -warp_count:] = video_visible_latents[:, :, -warp_count:]
                visible_history_mid = torch.zeros(batch_size, 1, history_sizes[1], h, w, device=device)
                visible_history_long = torch.zeros(batch_size, 1, history_sizes[0], h, w, device=device)

        long_size, mid_size, short_size = history_sizes
        latents_history_long = torch.zeros(batch_size, channels, long_size, h, w, device=device, dtype=torch.float32)
        latents_history_mid = torch.zeros(batch_size, channels, mid_size, h, w, device=device, dtype=torch.float32)
        latents_history_prev_short = torch.zeros(
            batch_size,
            channels,
            short_size,
            h,
            w,
            device=device,
            dtype=torch.float32,
        )
        fake_prev_short_count = 0
        if short_size > 0 and fake_image_latents is not None:
            fake_prev_short_count = min(int(short_size), int(fake_image_latents.shape[2]))
            latents_history_prev_short[:, :, -fake_prev_short_count:] = fake_image_latents[
                :, :, -fake_prev_short_count:
            ].to(device=device, dtype=torch.float32)
        latents_prefix = image_latents.to(device=device, dtype=torch.float32)
        latents_prefix = add_official_image_history_prefix_noise(latents_prefix, args, seq=seq)

        official_target_start = 1 + num_history_latents
        indices_hidden_states = torch.arange(
            official_target_start,
            official_target_start + latent_window_size,
            device=device,
            dtype=torch.long,
        )
        indices_latents_history_long = torch.arange(1, 1 + long_size, device=device, dtype=torch.long)
        indices_latents_history_mid = torch.arange(
            1 + long_size,
            1 + long_size + mid_size,
            device=device,
            dtype=torch.long,
        )
        indices_latents_history_prev_short = torch.arange(
            1 + long_size + mid_size,
            1 + long_size + mid_size + short_size,
            device=device,
            dtype=torch.long,
        )

        mode = getattr(args, "history_positioning", "none")
        delta = int(getattr(args, "history_position_delta", 0))
        if mode == "none":
            warp_start = official_target_start - latent_window_size
            indices_latents_history_short = torch.arange(
                warp_start,
                warp_start + latent_window_size,
                device=device,
                dtype=torch.long,
            )
        elif mode == "last_n":
            indices_latents_history_short = (
                official_target_start
                + torch.arange(latent_window_size - 1, -1, -1, device=device, dtype=torch.long)
                - delta
            )
        elif mode == "last_n_same_order":
            indices_latents_history_short = (
                official_target_start + torch.arange(latent_window_size, device=device, dtype=torch.long) - delta
            )
        else:
            raise ValueError(f"Unsupported --history_positioning: {mode}")
        if indices_latents_history_short.numel() and int(indices_latents_history_short.min().item()) < 0:
            raise ValueError(
                f"--history_position_delta={delta} makes negative history indices: "
                f"{indices_latents_history_short.tolist()}"
            )
        if mode != "none":
            print(
                json.dumps(
                    {
                        "event": "history_positioning",
                        "seq": seq,
                        "mode": mode,
                        "count": int(latent_window_size),
                        "delta": int(delta),
                        "video_count": int(video_frames),
                        "history_offset": 0,
                        "selected_history_positions": [0, int(latent_window_size)],
                        "target_start_index": int(official_target_start),
                        "old_indices": list(range(official_target_start - latent_window_size, official_target_start)),
                        "new_indices": indices_latents_history_short.detach().cpu().tolist(),
                    }
                ),
                flush=True,
            )

        prefix_index = torch.zeros(1, device=device, dtype=torch.long)
        if short_size > 0:
            latents_history_short = torch.cat([latents_prefix, latents_history_prev_short, warp_latents], dim=2)
            indices_latents_history_short = torch.cat(
                [prefix_index, indices_latents_history_prev_short, indices_latents_history_short],
                dim=0,
            )
            if visible_history_short is not None:
                prefix_visible = torch.ones(batch_size, 1, 1, h, w, device=device, dtype=torch.float32)
                prev_short_visible = torch.zeros(batch_size, 1, short_size, h, w, device=device, dtype=torch.float32)
                if fake_prev_short_count > 0:
                    prev_short_visible[:, :, -fake_prev_short_count:] = 1.0
                visible_history_short = torch.cat([prefix_visible, prev_short_visible, visible_history_short], dim=2)
        else:
            latents_history_short = torch.cat([latents_prefix, warp_latents], dim=2)
            indices_latents_history_short = torch.cat([prefix_index, indices_latents_history_short], dim=0)
            if visible_history_short is not None:
                prefix_visible = torch.ones(batch_size, 1, 1, h, w, device=device, dtype=torch.float32)
                visible_history_short = torch.cat([prefix_visible, visible_history_short], dim=2)
        item = {
            "event": "history_prepared",
            "history_sizes": history_sizes,
            "history_temporal_layout": "warp_as_history_official",
            "video_history_latent_frames": int(video_frames),
            "fake_image_history_latent_frames": int(fake_prev_short_count),
            "short_history_shape": list(latents_history_short.shape),
            "mid_history_shape": None if mid_size == 0 else list(latents_history_mid.shape),
            "long_history_shape": None if long_size == 0 else list(latents_history_long.shape),
            "indices_history_short": indices_latents_history_short.detach().cpu().tolist(),
            "indices_history_mid": indices_latents_history_mid.detach().cpu().tolist(),
            "indices_history_long": indices_latents_history_long.detach().cpu().tolist(),
            "indices_hidden_states": indices_hidden_states.detach().cpu().tolist(),
        }
        visibility_threshold = None
        visibility_mode = None
        if getattr(args, "history_visible_token_drop", False):
            visibility_threshold = float(args.history_visible_token_threshold)
            visibility_mode = "drop"
        elif using_history_invisible_token(args):
            visibility_threshold = float(args.history_invisible_token_threshold)
            visibility_mode = str(args.history_invisible_token_mode)
        if visibility_threshold is not None:
            short_keep = _estimate_patch_keep(visible_history_short, (1, 2, 2), visibility_threshold)
            mid_keep = _estimate_patch_keep(visible_history_mid, (2, 4, 4), visibility_threshold)
            long_keep = _estimate_patch_keep(visible_history_long, (4, 8, 8), visibility_threshold)
            item.update(
                {
                    "history_visibility_mode": visibility_mode,
                    "history_visible_token_threshold": visibility_threshold,
                    "history_short_keep_ratio_estimate": None
                    if short_keep is None
                    else float(short_keep.float().mean().cpu()),
                    "history_mid_keep_ratio_estimate": None
                    if mid_keep is None
                    else float(mid_keep.float().mean().cpu()),
                    "history_long_keep_ratio_estimate": None
                    if long_keep is None
                    else float(long_keep.float().mean().cpu()),
                    "history_short_keep_count_estimate": None
                    if short_keep is None
                    else int(short_keep.sum().item()),
                    "history_mid_keep_count_estimate": None if mid_keep is None else int(mid_keep.sum().item()),
                    "history_long_keep_count_estimate": None if long_keep is None else int(long_keep.sum().item()),
                }
            )
        print(json.dumps(item), flush=True)

        return {
            "indices_hidden_states": indices_hidden_states.unsqueeze(0),
            "indices_latents_history_short": indices_latents_history_short.unsqueeze(0),
            "indices_latents_history_mid": indices_latents_history_mid.unsqueeze(0) if mid_size > 0 else None,
            "indices_latents_history_long": indices_latents_history_long.unsqueeze(0) if long_size > 0 else None,
            "latents_history_short": latents_history_short.to(dtype=history_dtype),
            "latents_history_mid": latents_history_mid.to(dtype=history_dtype) if mid_size > 0 else None,
            "latents_history_long": latents_history_long.to(dtype=history_dtype) if long_size > 0 else None,
            "history_visible_mask_short": visible_history_short,
            "history_visible_mask_mid": visible_history_mid,
            "history_visible_mask_long": visible_history_long,
        }

    if video_latents is not None and num_history_latents > 0:
        video_latents = apply_history_video_latent_perturbation(video_latents, args, seq=seq)
        video_latents = add_official_video_history_noise(video_latents, args, seq=seq)
        history_frames = history_latents.shape[2]
        video_frames = video_latents.shape[2]
        video_count = min(int(video_frames), int(history_frames))
        history_offset = int(history_frames - video_count)
        if getattr(args, "history_visible_token_drop", False) or using_history_invisible_token(args):
            video_visible_latents = _load_video_visibility_latents(video_frames)
            if video_visible_latents is not None:
                history_visible_latents = torch.zeros(
                    batch_size, 1, num_history_latents, h, w, device=device, dtype=torch.float32
                )
                if video_frames < history_frames:
                    keep_frames = history_frames - video_frames
                    history_visible_latents = torch.cat(
                        [history_visible_latents[:, :, :keep_frames], video_visible_latents], dim=2
                    )
                else:
                    history_visible_latents = video_visible_latents[:, :, -history_frames:]
        if video_frames < history_frames:
            keep_frames = history_frames - video_frames
            history_latents = torch.cat([history_latents[:, :, :keep_frames], video_latents], dim=2)
        else:
            history_latents = video_latents[:, :, -history_frames:]
    elif fake_image_latents is not None:
        history_latents = torch.cat([history_latents, fake_image_latents], dim=2)
    indices = torch.arange(0, sum([1, *history_sizes, args.num_latent_frames_per_chunk]), device=device)
    (
        indices_prefix,
        indices_latents_history_long,
        indices_latents_history_mid,
        indices_latents_history_1x,
        indices_hidden_states,
    ) = indices.split([1, *history_sizes, args.num_latent_frames_per_chunk], dim=0)
    history_window = history_latents[:, :, -num_history_latents:] if num_history_latents > 0 else history_latents[:, :, :0]
    visible_window = (
        history_visible_latents[:, :, -num_history_latents:]
        if history_visible_latents is not None and num_history_latents > 0
        else None
    )
    assignment = getattr(args, "history_assignment", "contiguous")
    if assignment == "contiguous":
        layout = getattr(args, "history_temporal_layout", "long_mid_short")
        if layout == "long_mid_short":
            start = 0
            history_parts = []
            visible_parts = []
            for size in history_sizes:
                history_parts.append(history_window[:, :, start : start + size])
                if visible_window is not None:
                    visible_parts.append(visible_window[:, :, start : start + size])
                start += size
            latents_history_long, latents_history_mid, latents_history_1x = history_parts
            if visible_window is not None:
                visible_history_long, visible_history_mid, visible_history_1x = visible_parts
            else:
                visible_history_long = visible_history_mid = visible_history_1x = None
        elif layout == "short_mid_long":
            short_size = history_sizes[2]
            mid_size = history_sizes[1]
            long_size = history_sizes[0]
            cursor = 0
            latents_history_1x = history_window[:, :, cursor : cursor + short_size]
            visible_history_1x = (
                visible_window[:, :, cursor : cursor + short_size] if visible_window is not None else None
            )
            cursor += short_size
            latents_history_mid = history_window[:, :, cursor : cursor + mid_size]
            visible_history_mid = (
                visible_window[:, :, cursor : cursor + mid_size] if visible_window is not None else None
            )
            cursor += mid_size
            latents_history_long = history_window[:, :, cursor : cursor + long_size]
            visible_history_long = (
                visible_window[:, :, cursor : cursor + long_size] if visible_window is not None else None
            )
        else:
            raise ValueError(f"Unsupported --history_temporal_layout: {layout}")
        (
            indices_latents_history_long,
            indices_latents_history_mid,
            indices_latents_history_1x,
        ) = remap_history_rope_indices(
            indices_latents_history_long,
            indices_latents_history_mid,
            indices_latents_history_1x,
            indices_hidden_states,
            args,
            video_count=video_count,
            history_offset=history_offset,
            seq=seq,
        )
    elif assignment == "random_balanced":
        if video_latents is None or video_count <= 0:
            raise ValueError("--history_assignment random_balanced requires video history latents.")
        if video_count != num_history_latents:
            raise ValueError(
                "--history_assignment random_balanced currently requires sum(--history_sizes) to match the "
                f"actual video history latent count. Got sum={num_history_latents}, video_count={video_count}."
            )
        if any(size <= 0 for size in history_sizes):
            raise ValueError("--history_assignment random_balanced requires all three history sizes to be > 0.")
        if getattr(args, "history_positioning", "none") != "none":
            raise ValueError("--history_assignment random_balanced cannot be combined with history positioning.")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(stable_seed_from_parts(args.history_assignment_seed, seq or "unknown", *history_sizes))
        order = torch.randperm(video_count, generator=generator)
        base_history_indices = torch.cat(
            [indices_latents_history_long, indices_latents_history_mid, indices_latents_history_1x], dim=0
        )
        actual_history_indices = base_history_indices[history_offset : history_offset + video_count]

        random_parts = []
        random_indices = []
        random_visible_parts = []
        selected_by_bucket = {}
        cursor = 0
        for name, size in zip(("long", "mid", "short"), history_sizes):
            selected = order[cursor : cursor + size].sort().values.to(device=device)
            cursor += size
            random_parts.append(history_window.index_select(2, history_offset + selected))
            random_indices.append(actual_history_indices.index_select(0, selected))
            if visible_window is not None:
                random_visible_parts.append(visible_window.index_select(2, history_offset + selected))
            selected_by_bucket[name] = selected.detach().cpu().tolist()

        latents_history_long, latents_history_mid, latents_history_1x = random_parts
        indices_latents_history_long, indices_latents_history_mid, indices_latents_history_1x = random_indices
        if visible_window is not None:
            visible_history_long, visible_history_mid, visible_history_1x = random_visible_parts
        else:
            visible_history_long = visible_history_mid = visible_history_1x = None
        print(
            json.dumps(
                {
                    "event": "history_assignment",
                    "seq": seq,
                    "mode": assignment,
                    "temporal_layout": getattr(args, "history_temporal_layout", "long_mid_short"),
                    "seed": int(args.history_assignment_seed),
                    "history_sizes": history_sizes,
                    "selected_frame_positions": selected_by_bucket,
                    "indices_history_long": indices_latents_history_long.detach().cpu().tolist(),
                    "indices_history_mid": indices_latents_history_mid.detach().cpu().tolist(),
                    "indices_history_short": indices_latents_history_1x.detach().cpu().tolist(),
                }
            ),
            flush=True,
        )
    else:
        raise ValueError(f"Unsupported --history_assignment: {assignment}")
    if bool(getattr(args, "use_warp_as_history", False)):
        latents_history_short = latents_history_1x
        indices_latents_history_short = indices_latents_history_1x
        visible_history_short = visible_history_1x
    else:
        latents_history_short = torch.cat([image_latents, latents_history_1x], dim=2)
        indices_latents_history_short = torch.cat([indices_prefix, indices_latents_history_1x], dim=0)
        if visible_history_1x is not None:
            prefix_visible = torch.ones(batch_size, 1, image_latents.shape[2], h, w, device=device, dtype=torch.float32)
            visible_history_short = torch.cat([prefix_visible, visible_history_1x], dim=2)
        else:
            visible_history_short = None

    def _maybe_history_tensor(tensor):
        if tensor.shape[2] == 0:
            return None
        return tensor.to(dtype=history_dtype)

    def _maybe_indices(tensor):
        if tensor.numel() == 0:
            return None
        return tensor.unsqueeze(0)

    def _maybe_mask_tensor(tensor):
        if tensor is None or tensor.shape[2] == 0:
            return None
        return tensor.to(dtype=torch.float32)

    item = {
        "event": "history_prepared",
        "history_sizes": history_sizes,
        "history_temporal_layout": getattr(args, "history_temporal_layout", "long_mid_short"),
        "video_history_latent_frames": 0 if video_latents is None else int(video_latents.shape[2]),
        "short_history_shape": list(latents_history_short.shape),
        "mid_history_shape": None if latents_history_mid.shape[2] == 0 else list(latents_history_mid.shape),
        "long_history_shape": None if latents_history_long.shape[2] == 0 else list(latents_history_long.shape),
        "indices_history_short": indices_latents_history_short.detach().cpu().tolist(),
        "indices_history_mid": indices_latents_history_mid.detach().cpu().tolist(),
        "indices_history_long": indices_latents_history_long.detach().cpu().tolist(),
        "indices_hidden_states": indices_hidden_states.detach().cpu().tolist(),
    }
    visibility_threshold = None
    visibility_mode = None
    if getattr(args, "history_visible_token_drop", False):
        visibility_threshold = float(args.history_visible_token_threshold)
        visibility_mode = "drop"
    elif using_history_invisible_token(args):
        visibility_threshold = float(args.history_invisible_token_threshold)
        visibility_mode = str(args.history_invisible_token_mode)
    if visibility_threshold is not None:
        short_keep = _estimate_patch_keep(visible_history_short, (1, 2, 2), visibility_threshold)
        mid_keep = _estimate_patch_keep(visible_history_mid, (2, 4, 4), visibility_threshold)
        long_keep = _estimate_patch_keep(visible_history_long, (4, 8, 8), visibility_threshold)
        item.update(
            {
                "history_visibility_mode": visibility_mode,
                "history_visible_token_threshold": visibility_threshold,
                "history_short_keep_ratio_estimate": None
                if short_keep is None
                else float(short_keep.float().mean().cpu()),
                "history_mid_keep_ratio_estimate": None
                if mid_keep is None
                else float(mid_keep.float().mean().cpu()),
                "history_long_keep_ratio_estimate": None
                if long_keep is None
                else float(long_keep.float().mean().cpu()),
                "history_short_keep_count_estimate": None
                if short_keep is None
                else int(short_keep.sum().item()),
                "history_mid_keep_count_estimate": None if mid_keep is None else int(mid_keep.sum().item()),
                "history_long_keep_count_estimate": None if long_keep is None else int(long_keep.sum().item()),
            }
        )
    print(json.dumps(item), flush=True)

    return {
        "indices_hidden_states": indices_hidden_states.unsqueeze(0),
        "indices_latents_history_short": indices_latents_history_short.unsqueeze(0),
        "indices_latents_history_mid": _maybe_indices(indices_latents_history_mid),
        "indices_latents_history_long": _maybe_indices(indices_latents_history_long),
        "latents_history_short": latents_history_short.to(dtype=history_dtype),
        "latents_history_mid": _maybe_history_tensor(latents_history_mid),
        "latents_history_long": _maybe_history_tensor(latents_history_long),
        "history_visible_mask_short": _maybe_mask_tensor(visible_history_short),
        "history_visible_mask_mid": _maybe_mask_tensor(visible_history_mid),
        "history_visible_mask_long": _maybe_mask_tensor(visible_history_long),
    }

def using_history_invisible_token(args):
    return str(getattr(args, "history_invisible_token_mode", "none") or "none") != "none"

def transformer_compute_dtype(transformer):
    condition_embedder = getattr(transformer, "condition_embedder", None)
    time_proj = getattr(condition_embedder, "time_proj", None)
    time_proj_weight = getattr(time_proj, "weight", None)
    if torch.is_tensor(time_proj_weight) and torch.is_floating_point(time_proj_weight):
        return time_proj_weight.dtype

    for module_name in ("patch_embedding", "patch_short", "patch_mid", "patch_long"):
        module = getattr(transformer, module_name, None)
        weight = getattr(module, "weight", None)
        if torch.is_tensor(weight) and torch.is_floating_point(weight):
            return weight.dtype

    for name, param in transformer.named_parameters():
        if not torch.is_floating_point(param):
            continue
        if name == "history_invisible_token" or ".lora_" in name:
            continue
        return param.dtype

    for _name, param in transformer.named_parameters():
        if torch.is_floating_point(param):
            return param.dtype
    return torch.float32

def ensure_history_invisible_token(transformer):
    param = getattr(transformer, "history_invisible_token", None)
    if param is not None:
        return param
    inner_dim = getattr(transformer, "inner_dim", None)
    if inner_dim is None:
        raise ValueError("Transformer missing inner_dim; cannot initialize history invisible token.")
    ref_param = next(transformer.parameters())
    token = torch.nn.Parameter(torch.zeros(1, 1, int(inner_dim), device=ref_param.device, dtype=ref_param.dtype))
    torch.nn.init.normal_(token, mean=0.0, std=0.02)
    transformer.register_parameter("history_invisible_token", token)
    return transformer.history_invisible_token

def visible_aux_state_dict(transformer):
    state = {}
    token = getattr(transformer, "history_invisible_token", None)
    if token is not None:
        state["history_invisible_token"] = token.detach().cpu()
    return state

def downsample_latents_spatial_bilinear(latents, height, width, scale=1.0):
    batch_size, channels, latent_frames, _, _ = latents.shape
    latents_2d = latents.permute(0, 2, 1, 3, 4).reshape(
        batch_size * latent_frames, channels, latents.shape[-2], latents.shape[-1]
    )
    latents_2d = F.interpolate(latents_2d, size=(height, width), mode="bilinear")
    latents_2d = latents_2d * float(scale)
    return latents_2d.reshape(batch_size, latent_frames, channels, height, width).permute(0, 2, 1, 3, 4)

def training_exact_dynamic_shifting_enabled(pipe, args):
    if args.flow_matching_use_dynamic_shifting == "on":
        return True
    if args.flow_matching_use_dynamic_shifting == "off":
        return False
    return bool(pipe.scheduler.config.get("use_dynamic_shifting", False))

def training_exact_pyramid_latents(target_latents, pyramid_stage_num):
    pyramid_latent_list = [target_latents.float()]
    latents = target_latents.float()
    height, width = latents.shape[-2], latents.shape[-1]
    for _ in range(pyramid_stage_num - 1):
        height //= 2
        width //= 2
        latents = downsample_latents_spatial_bilinear(latents, height, width)
        pyramid_latent_list.append(latents)
    return list(reversed(pyramid_latent_list))

def training_exact_pyramid_noises(reference_latents, pyramid_stage_num):
    noise = torch.randn_like(reference_latents)
    noise_list = [noise]
    cur_noise = noise
    height, width = noise.shape[-2], noise.shape[-1]
    for _ in range(pyramid_stage_num - 1):
        height //= 2
        width //= 2
        cur_noise = downsample_latents_spatial_bilinear(cur_noise, height, width, scale=2.0)
        noise_list.append(cur_noise)
    return list(reversed(noise_list))

def flow_matching_train_exact_items(pipe, target_latents, args, device):
    pyramid_stage_num = len(args.pyramid_num_inference_steps_list)
    scheduler_stages = int(pipe.scheduler.config.get("stages", pyramid_stage_num))
    if pyramid_stage_num != scheduler_stages:
        raise ValueError(
            f"--flow_matching_mode train_exact expects {scheduler_stages} stages from the scheduler, "
            f"got {pyramid_stage_num} from --pyramid_num_inference_steps_list."
        )

    pyramid_latent_list = training_exact_pyramid_latents(target_latents, pyramid_stage_num)
    noise_list = training_exact_pyramid_noises(pyramid_latent_list[-1], pyramid_stage_num)
    training_steps = int(pipe.scheduler.config.num_train_timesteps)
    use_dynamic_shifting = training_exact_dynamic_shifting_enabled(pipe, args)
    timestep_sampling = str(getattr(args, "flow_matching_train_exact_timestep_sampling", "training_density"))
    items = []
    if args.flow_matching_stage_sampling == "fixed":
        if args.flow_matching_stage_id < 0:
            stage_ids = list(range(pyramid_stage_num))
        else:
            stage_ids = [int(args.flow_matching_stage_id)]
    else:
        stage_ids = list(range(pyramid_stage_num))

    for stage_id in stage_ids:
        clean_latent = pyramid_latent_list[stage_id]
        start_sigma = pipe.scheduler.start_sigmas[stage_id]
        end_sigma = pipe.scheduler.end_sigmas[stage_id]

        if stage_id == 0:
            start_point = noise_list[stage_id]
        else:
            last_clean_latent = pyramid_latent_list[stage_id - 1]
            last_clean_latent = upsample_stage_latents(last_clean_latent, clean_latent)
            start_point = start_sigma * noise_list[stage_id] + (1 - start_sigma) * last_clean_latent

        if stage_id == pyramid_stage_num - 1:
            end_point = clean_latent
        else:
            end_point = end_sigma * noise_list[stage_id] + (1 - end_sigma) * clean_latent

        if timestep_sampling == "training_density":
            u = compute_density_for_timestep_sampling(
                weighting_scheme=args.weighting_scheme,
                batch_size=target_latents.shape[0],
                logit_mean=0.0,
                logit_std=1.0,
                mode_scale=1.29,
            )
            indices = (u * training_steps).long().clamp(0, training_steps - 1).detach().cpu()
            timesteps = pipe.scheduler.timesteps_per_stage[stage_id][indices].to(device=device)
            sigmas = pipe.scheduler.sigmas_per_stage[stage_id][indices].to(device=device, dtype=start_point.dtype)
        elif timestep_sampling == "first":
            indices = torch.zeros((target_latents.shape[0],), dtype=torch.long)
            timesteps = pipe.scheduler.timesteps_per_stage[stage_id][indices].to(device=device)
            sigmas = pipe.scheduler.sigmas_per_stage[stage_id][indices].to(device=device, dtype=start_point.dtype)
        elif timestep_sampling == "first_second_interval":
            patch_size = pipe.transformer.config.patch_size
            image_seq_len = (clean_latent.shape[-1] * clean_latent.shape[-2] * clean_latent.shape[-3]) // (
                patch_size[0] * patch_size[1] * patch_size[2]
            )
            mu = calculate_shift(
                image_seq_len,
                pipe.scheduler.config.get("base_image_seq_len", 256),
                pipe.scheduler.config.get("max_image_seq_len", 4096),
                pipe.scheduler.config.get("base_shift", 0.5),
                pipe.scheduler.config.get("max_shift", 1.15),
            )
            pipe.scheduler.set_timesteps(
                args.pyramid_num_inference_steps_list[stage_id],
                stage_id,
                device=device,
                mu=mu,
                is_amplify_first_chunk=bool(args.is_amplify_first_chunk),
            )
            sched_timesteps = pipe.scheduler.timesteps.to(device=device)
            sched_sigmas = pipe.scheduler.sigmas.to(device=device, dtype=start_point.dtype)
            if sched_timesteps.numel() < 2 or sched_sigmas.numel() < 2:
                raise ValueError(
                    "--flow_matching_train_exact_timestep_sampling first_second_interval requires at least 2 "
                    "inference steps on the selected stage."
                )
            alpha = torch.rand((target_latents.shape[0],), device=device, dtype=start_point.dtype)
            t0 = sched_timesteps[0].to(dtype=start_point.dtype)
            t1 = sched_timesteps[1].to(dtype=start_point.dtype)
            s0 = sched_sigmas[0]
            s1 = sched_sigmas[1]
            timesteps = t0 + alpha * (t1 - t0)
            sigmas = s0 + alpha * (s1 - s0)
            indices = torch.zeros((target_latents.shape[0],), dtype=torch.long)
        else:
            raise ValueError(
                f"Unsupported --flow_matching_train_exact_timestep_sampling: {timestep_sampling}"
            )
        while sigmas.ndim < start_point.ndim:
            sigmas = sigmas.unsqueeze(-1)

        if use_dynamic_shifting:
            sigmas = apply_schedule_shift(
                sigmas,
                start_point,
                base_seq_len=256,
                max_seq_len=4096,
                base_shift=0.5,
                max_shift=1.15,
                time_shift_type=pipe.scheduler.config.get("time_shift_type", "linear"),
            )
            stage_timesteps = pipe.scheduler.timesteps_per_stage[stage_id].to(device=device, dtype=sigmas.dtype)
            timesteps = stage_timesteps.min() + sigmas * (stage_timesteps.max() - stage_timesteps.min())
            while timesteps.ndim > 1:
                timesteps = timesteps.squeeze(-1)

        noisy_latents = sigmas * start_point + (1 - sigmas) * end_point
        target = start_point - end_point
        items.append(
            {
                "stage_id": stage_id,
                "noisy_latents": noisy_latents,
                "sigmas": sigmas,
                "timesteps": timesteps,
                "target": target,
                "start_point": start_point,
                "end_point": end_point,
                "indices": indices,
                "use_dynamic_shifting": use_dynamic_shifting,
            }
        )

    return items

def flow_matching_loss_train_exact(
    pipe,
    prompt_embeds,
    target_latents,
    histories,
    args,
    device,
):
    stage_items = flow_matching_train_exact_items(pipe, target_latents, args, device)
    stage_losses = []
    stage_ids = [int(item["stage_id"]) for item in stage_items]
    stats = {
        "flow_matching_mode": "train_exact",
        "flow_matching_stage_id": int(args.flow_matching_stage_id),
        "flow_matching_stage_sampling": args.flow_matching_stage_sampling,
        "flow_matching_timestep_sampling": str(
            getattr(args, "flow_matching_train_exact_timestep_sampling", "training_density")
        ),
        "flow_matching_train_exact_stage_count": len(stage_items),
        "flow_matching_train_exact_stage_ids": stage_ids,
        "flow_matching_use_dynamic_shifting": bool(stage_items[0]["use_dynamic_shifting"]) if stage_items else False,
        "flow_matching_navit_forward": True,
    }

    transformer_dtype = transformer_compute_dtype(pipe.transformer)
    noisy_model_inputs = [item["noisy_latents"].to(dtype=transformer_dtype) for item in stage_items]
    timesteps = [item["timesteps"] for item in stage_items]
    targets = [item["target"] for item in stage_items]
    sigmas = [item["sigmas"] for item in stage_items]

    model_pred = transformer_model_forward(
        pipe,
        noisy_model_inputs,
        timesteps,
        prompt_embeds,
        histories,
        attention_kwargs=None,
        target_channel_fusion_latents=None,
        is_first_denoising_step=False,
    )
    if not isinstance(model_pred, list):
        raise TypeError(
            "--flow_matching_mode train_exact expects a NaViT list prediction. "
            "Use the training_modules transformer implementation for Helios stage2-post parity."
        )
    if len(model_pred) != len(stage_items):
        raise ValueError(f"NaViT prediction count mismatch: got {len(model_pred)}, expected {len(stage_items)}.")

    for item, cur_model_pred, target, cur_sigmas in zip(stage_items, model_pred, targets, sigmas):
        stage_id = item["stage_id"]
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=cur_sigmas)
        stage_loss = torch.mean(
            (weighting.float() * (cur_model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        ).mean()
        stage_losses.append(stage_loss)
        stats[f"flow_mse_stage{stage_id}"] = stage_loss.detach()
        stats[f"target_flow_norm_stage{stage_id}"] = target.float().square().mean().sqrt().detach()
        stats[f"pred_flow_norm_stage{stage_id}"] = cur_model_pred.float().square().mean().sqrt().detach()
        stats[f"sigma_stage{stage_id}"] = cur_sigmas.detach().float().mean()
        stats[f"timestep_stage{stage_id}"] = item["timesteps"].detach().float().mean()
        stats[f"timestep_index_stage{stage_id}"] = item["indices"].float().mean()

    # Helios stage2 post uses is_navit_pyramid=true: all pyramid stages are fed
    # to the transformer as a single list forward, then _flow_loss averages them.
    total_loss = torch.stack(stage_losses).mean()
    stats["flow_mse"] = total_loss.detach()
    stats["flow_mse_mean_stage"] = torch.stack([loss.detach() for loss in stage_losses]).mean()
    return total_loss, stats, None

def flow_matching_loss(
    pipe,
    prompt_embeds,
    target_latents,
    histories,
    args,
    device,
):
    return flow_matching_loss_train_exact(
        pipe,
        prompt_embeds,
        target_latents,
        histories,
        args,
        device,
    )

def lora_target_modules(args):
    return [name.strip() for name in args.lora_target_modules.split(",") if name.strip()]

def setup_visible_lora(transformer, args, seq):
    adapter_name = f"{args.lora_adapter_name}_{stable_seed_from_parts(args.seed, seq) % 10_000_000}"
    try:
        transformer.delete_adapters(adapter_name)
    except Exception:
        pass
    transformer.requires_grad_(False)
    config = LoraConfig(
        r=int(args.lora_rank),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        init_lora_weights=True,
        target_modules=lora_target_modules(args),
    )
    transformer.add_adapter(config, adapter_name=adapter_name)
    transformer.set_adapter(adapter_name)
    optimize_lora_now = int(getattr(args, "iters", 0)) > 0
    base_dtype = transformer_compute_dtype(transformer)
    if optimize_lora_now:
        transformer.train()
    else:
        transformer.eval()
    trainable = []
    trainable_ids = set()
    if using_history_invisible_token(args):
        token = ensure_history_invisible_token(transformer)
        token.requires_grad_(optimize_lora_now)
        if optimize_lora_now:
            token.data = token.data.float()
            trainable.append(token)
            trainable_ids.add(id(token))
        else:
            token.data = token.data.to(dtype=base_dtype)
    for name, param in transformer.named_parameters():
        if param.requires_grad:
            if optimize_lora_now:
                param.data = param.data.float()
            if id(param) not in trainable_ids:
                trainable.append(param)
                trainable_ids.add(id(param))
    if not trainable:
        raise ValueError("No trainable LoRA parameters found; check --lora_target_modules.")
    stats = {
        "event": "visible_lora_setup",
        "seq": seq,
        "adapter_name": adapter_name,
        "lora_rank": int(args.lora_rank),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "lora_target_modules": lora_target_modules(args),
        "lora_trainable_params": int(sum(p.numel() for p in trainable)),
        "history_invisible_token_mode": str(getattr(args, "history_invisible_token_mode", "none") or "none"),
    }
    return adapter_name, trainable, stats

def save_visible_lora_state(transformer, out_dir, adapter_name, filename="visible_lora_state.pt"):
    state = get_peft_model_state_dict(transformer, adapter_name=adapter_name)
    payload = {
        "peft": {key: value.detach().cpu() for key, value in state.items()},
        "extra_state": visible_aux_state_dict(transformer),
    }
    torch.save(payload, Path(out_dir) / filename)

def schedule_multiplier(step, total_steps, schedule, final_ratio):
    if total_steps <= 1 or schedule == "constant":
        return 1.0
    progress = float(step) / float(max(total_steps - 1, 1))
    final_ratio = float(final_ratio)
    if schedule == "linear":
        return final_ratio + (1.0 - final_ratio) * (1.0 - progress)
    if schedule == "cosine":
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_ratio + (1.0 - final_ratio) * cosine
    raise ValueError(f"Unsupported schedule {schedule}.")

def lr_schedule_multiplier(step, total_steps, args):
    return schedule_multiplier(step, total_steps, args.lr_schedule, args.lr_schedule_final_ratio)

def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr

def transformer_model_forward(
    pipe,
    latents,
    timestep,
    encoder_hidden_states,
    histories,
    attention_kwargs,
    target_channel_fusion_latents=None,
    is_first_denoising_step=False,
):
    return pipe.transformer(
        hidden_states=latents,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        attention_kwargs=attention_kwargs,
        return_dict=False,
        indices_hidden_states=histories["indices_hidden_states"],
        indices_latents_history_short=histories["indices_latents_history_short"],
        indices_latents_history_mid=histories["indices_latents_history_mid"],
        indices_latents_history_long=histories["indices_latents_history_long"],
        latents_history_short=histories["latents_history_short"],
        latents_history_mid=histories["latents_history_mid"],
        latents_history_long=histories["latents_history_long"],
        history_visible_mask_short=histories.get("history_visible_mask_short"),
        history_visible_mask_mid=histories.get("history_visible_mask_mid"),
        history_visible_mask_long=histories.get("history_visible_mask_long"),
        target_channel_fusion_latents=target_channel_fusion_latents,
    )[0]

def upsample_stage_latents(latents, reference):
    if latents.shape[2:] == reference.shape[2:]:
        return latents
    batch_size, channels, latent_frames, _, _ = latents.shape
    height, width = reference.shape[-2:]
    latents_2d = latents.permute(0, 2, 1, 3, 4).reshape(
        batch_size * latent_frames, channels, latents.shape[-2], latents.shape[-1]
    )
    latents_2d = F.interpolate(latents_2d.float(), size=(height, width), mode="nearest")
    return latents_2d.reshape(batch_size, latent_frames, channels, height, width).permute(0, 2, 1, 3, 4)

def validate_args(args):
    if int(args.num_frames) <= 0:
        raise ValueError("--num_frames must be positive.")
    if int(args.num_latent_frames_per_chunk) <= 0:
        raise ValueError("--num_latent_frames_per_chunk must be positive.")
    if len(args.history_sizes) != 3:
        raise ValueError("--history_sizes must contain three values.")
    if any(int(x) < 0 for x in args.history_sizes):
        raise ValueError("--history_sizes values must be non-negative.")
    if args.flow_matching_stage_sampling == "fixed" and int(args.flow_matching_stage_id) < 0:
        raise ValueError("--flow_matching_stage_id must be non-negative when stage sampling is fixed.")
    if int(args.history_position_count) < 0:
        raise ValueError("--history_position_count must be non-negative.")
    if int(args.history_position_delta) < 0:
        raise ValueError("--history_position_delta must be non-negative.")
    for prefix in ("image", "video"):
        sigma_min = float(getattr(args, f"{prefix}_noise_sigma_min", 0.111))
        sigma_max = float(getattr(args, f"{prefix}_noise_sigma_max", 0.135))
        if sigma_min < 0.0 or sigma_max < sigma_min:
            raise ValueError(
                f"--{prefix}_noise_sigma_min/max must satisfy 0 <= min <= max, got [{sigma_min}, {sigma_max}]."
            )
