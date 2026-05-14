#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("XFORMERS_DISABLED", "1")


def _disable_broken_flash_attn_imports():
    try:
        import transformers.utils as transformers_utils
        import transformers.utils.import_utils as import_utils
    except Exception:
        return

    def _false(*_args, **_kwargs):
        return False

    import_utils.is_flash_attn_2_available = _false
    import_utils.is_flash_attn_greater_or_equal = _false
    import_utils.is_flash_attn_greater_or_equal_2_10 = _false
    transformers_utils.is_flash_attn_2_available = _false
    transformers_utils.is_flash_attn_greater_or_equal = _false
    transformers_utils.is_flash_attn_greater_or_equal_2_10 = _false
    try:
        import diffusers.utils as diffusers_utils
        import diffusers.utils.import_utils as diffusers_import_utils

        diffusers_import_utils._xformers_available = False
        diffusers_import_utils._flash_attn_available = False
        diffusers_import_utils._flash_attn_3_available = False
        diffusers_import_utils.is_xformers_available = _false
        diffusers_import_utils.is_flash_attn_available = _false
        diffusers_import_utils.is_flash_attn_3_available = _false
        diffusers_import_utils.is_flash_attn_version = _false
        diffusers_utils.is_xformers_available = _false
        diffusers_utils.is_flash_attn_available = _false
        diffusers_utils.is_flash_attn_3_available = _false
        diffusers_utils.is_flash_attn_version = _false
    except Exception:
        pass


_disable_broken_flash_attn_imports()

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import torch
from tqdm import tqdm

from warp_as_history.training import core as opt
from warp_as_history.training.data import (
    LazyPreparedItems,
    build_online_warp_training_cache,
    normalize_online_training_dataframe,
)
from warp_as_history.training.utils import (
    current_train_lr,
    release_cuda_cache,
    save_lora,
    scalar,
    write_json,
    next_index_generator,
)

DEFAULT_HELIOS_MODEL = "checkpoints/helios-distilled"


def checkpoint_model_path(value, *, label):
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = Path(str(path.absolute()))
    checkpoints_root = Path(str((REPO_ROOT / "checkpoints").absolute()))
    if not path.is_relative_to(checkpoints_root):
        raise ValueError(f"{label} must be under {checkpoints_root}, got {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {label} directory: {path}. Run `python scripts/check_models.py`.")
    return str(path)


def _json_text(obj):
    return "```json\n" + json.dumps(obj, indent=2, sort_keys=True) + "\n```"


def create_tensorboard_writer(args, out_dir):
    if not bool(args.tensorboard):
        return None, None
    log_dir = Path(args.tensorboard_log_dir) if args.tensorboard_log_dir else out_dir / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorBoard logging requires the tensorboard package. Install requirements.txt or run `pip install tensorboard`."
        ) from exc
    writer = SummaryWriter(log_dir=str(log_dir))
    print(json.dumps({"event": "tensorboard_enabled", "log_dir": str(log_dir)}), flush=True)
    return writer, log_dir


def tensorboard_add_scalar(writer, tag, value, step):
    if writer is None:
        return
    if isinstance(value, bool):
        value = float(value)
    if isinstance(value, (int, float)):
        value = float(value)
        if math.isfinite(value):
            writer.add_scalar(tag, value, int(step))


def tensorboard_log_record(writer, record, step):
    if writer is None:
        return
    tensorboard_add_scalar(writer, "train/loss", record.get("loss"), step)
    tensorboard_add_scalar(writer, "train/lr", record.get("lr"), step)
    tensorboard_add_scalar(writer, "train/grad_norm", record.get("grad_norm"), step)
    tensorboard_add_scalar(writer, "train/elapsed_s", record.get("elapsed_s"), step)
    skip = {
        "step",
        "seq",
        "loss",
        "lr",
        "optimizer",
        "adamw_weight_decay",
        "warmup_steps",
        "max_grad_norm",
        "lora_rank",
        "lora_alpha",
        "lora_target_modules",
    }
    for key, value in record.items():
        if key in skip:
            continue
        tensorboard_add_scalar(writer, f"stats/{key}", value, step)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the release Warp-as-History LoRA.")
    parser.add_argument("--base_model_path", default=DEFAULT_HELIOS_MODEL)
    parser.add_argument(
        "--transformer_path",
        default="",
        help="Optional transformer-only checkpoint. Defaults to --base_model_path.",
    )
    parser.add_argument("--data_root", default="data/training")
    parser.add_argument("--prompt_csv", default="data/training/training_data.csv")
    parser.add_argument("--output_dir", default="runs/warp_as_history_lora")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_schedule", choices=["constant", "cosine", "linear"], default="constant")
    parser.add_argument("--lr_schedule_final_ratio", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num_frames", type=int, default=33)
    parser.add_argument("--num_latent_frames_per_chunk", type=int, default=9)
    parser.add_argument("--history_sizes", type=int, nargs=3, default=[16, 2, 1])
    parser.add_argument("--history_temporal_layout", choices=["long_mid_short", "short_mid_long"], default="long_mid_short")
    parser.add_argument("--pyramid_num_inference_steps_list", type=int, nargs="+", default=[2, 2, 2])
    parser.add_argument("--attention_backend", default="native")
    parser.add_argument("--flow_matching_stage_sampling", choices=["all", "fixed"], default="fixed")
    parser.add_argument("--flow_matching_stage_id", type=int, default=0)
    parser.add_argument("--flow_matching_train_exact_timestep_sampling", choices=["training_density", "first", "first_second_interval"], default="training_density")
    parser.add_argument("--history_positioning", choices=["none", "last_n", "last_n_same_order"], default="last_n_same_order")
    parser.add_argument("--history_position_count", type=int, default=9)
    parser.add_argument("--history_position_delta", type=int, default=0)
    parser.add_argument("--add_noise_to_video_latents", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image_noise_sigma_min", type=float, default=0.111)
    parser.add_argument("--image_noise_sigma_max", type=float, default=0.135)
    parser.add_argument("--video_noise_sigma_min", type=float, default=0.111)
    parser.add_argument("--video_noise_sigma_max", type=float, default=0.135)
    parser.add_argument("--visible_token_drop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visible_token_threshold", type=float, default=0.1)
    parser.add_argument("--direction_augmentation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--direction_reverse_probability", type=float, default=0.5)
    parser.add_argument("--online_video_column", default="")
    parser.add_argument("--online_prompt_column", default="")
    parser.add_argument("--online_prompt_trigger", default="camctl23x.")
    parser.add_argument("--online_frame_stride", type=int, default=1)
    parser.add_argument("--online_max_video_frames", type=int, default=0)
    parser.add_argument("--online_first_chunk_prob", type=float, default=0.5)
    parser.add_argument("--online_max_history_frames", type=int, default=19)
    parser.add_argument("--online_future_keyframe_prob", type=float, default=0.5)
    parser.add_argument("--online_future_keyframes_min", type=int, default=1)
    parser.add_argument("--online_future_keyframes_max", type=int, default=2)
    parser.add_argument("--online_pi3_pixel_limit", type=int, default=255000)
    parser.add_argument("--online_pi3_conf_threshold", type=float, default=0.1)
    parser.add_argument("--online_pi3_depth_edge_rtol", type=float, default=0.03)
    parser.add_argument("--online_mesh_samples_per_axis", type=int, default=4)
    parser.add_argument("--online_render_mode", default="target_fill", choices=["splat", "target_fill"])
    parser.add_argument("--online_target_fill_radius", type=int, default=1)
    parser.add_argument("--online_target_fill_min_neighbors", type=int, default=4)
    parser.add_argument("--online_mesh_break_mode", default="depth_normal")
    parser.add_argument("--online_mesh_depth_rtol", type=float, default=0.03)
    parser.add_argument("--online_mesh_normal_tol_deg", type=float, default=5.0)
    parser.add_argument("--online_invisible_fill", default="mean_first_frame", choices=["mean_first_frame", "black"])
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", default="attn1.to_q,attn1.to_k,attn1.to_v,attn1.to_out.0")
    parser.add_argument("--lora_adapter_name", default="warp_as_history")
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tensorboard_log_dir", default="")
    parser.add_argument("--prompt_cache_dir", default="data/training/prompt_cache/helios_distilled_512")
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_exact_args(args):
    exact = opt.parse_args([])
    exact.base_model_path = checkpoint_model_path(args.base_model_path, label="--base_model_path")
    exact.transformer_path = checkpoint_model_path(
        args.transformer_path or args.base_model_path,
        label="--transformer_path",
    )
    exact.data_root = args.data_root
    exact.prompt_csv = args.prompt_csv
    exact.output_dir = args.output_dir
    exact.limit = args.limit
    exact.use_warp_as_history = True
    exact.history_positioning = args.history_positioning
    exact.history_position_count = int(args.history_position_count)
    exact.history_position_delta = int(args.history_position_delta)
    exact.add_noise_to_video_latents = bool(args.add_noise_to_video_latents)
    exact.image_noise_sigma_min = float(args.image_noise_sigma_min)
    exact.image_noise_sigma_max = float(args.image_noise_sigma_max)
    exact.video_noise_sigma_min = float(args.video_noise_sigma_min)
    exact.video_noise_sigma_max = float(args.video_noise_sigma_max)
    exact.history_visible_token_drop = bool(args.visible_token_drop)
    exact.history_visible_token_threshold = float(args.visible_token_threshold)
    exact.history_invisible_token_mode = "none"
    exact.history_invisible_token_threshold = float(args.visible_token_threshold)
    exact.online_video_column = str(args.online_video_column)
    exact.online_prompt_column = str(args.online_prompt_column)
    exact.online_prompt_trigger = str(args.online_prompt_trigger)
    exact.online_direction_augmentation = bool(args.direction_augmentation)
    exact.online_direction_reverse_prob = float(args.direction_reverse_probability)
    exact.online_frame_stride = int(args.online_frame_stride)
    exact.online_max_video_frames = int(args.online_max_video_frames)
    exact.online_first_chunk_prob = float(args.online_first_chunk_prob)
    exact.online_max_history_frames = int(args.online_max_history_frames)
    exact.online_future_keyframe_prob = float(args.online_future_keyframe_prob)
    exact.online_future_keyframes_min = int(args.online_future_keyframes_min)
    exact.online_future_keyframes_max = int(args.online_future_keyframes_max)
    exact.online_pi3_pixel_limit = int(args.online_pi3_pixel_limit)
    exact.online_pi3_conf_threshold = float(args.online_pi3_conf_threshold)
    exact.online_pi3_depth_edge_rtol = float(args.online_pi3_depth_edge_rtol)
    exact.online_mesh_samples_per_axis = int(args.online_mesh_samples_per_axis)
    exact.online_render_mode = str(args.online_render_mode)
    exact.online_target_fill_radius = int(args.online_target_fill_radius)
    exact.online_target_fill_min_neighbors = int(args.online_target_fill_min_neighbors)
    exact.online_mesh_break_mode = str(args.online_mesh_break_mode)
    exact.online_mesh_depth_rtol = float(args.online_mesh_depth_rtol)
    exact.online_mesh_normal_tol_deg = float(args.online_mesh_normal_tol_deg)
    exact.online_invisible_fill = str(args.online_invisible_fill)
    exact.height = int(args.height)
    exact.width = int(args.width)
    exact.num_frames = int(args.num_frames)
    exact.num_latent_frames_per_chunk = int(args.num_latent_frames_per_chunk)
    exact.history_sizes = [int(x) for x in args.history_sizes]
    exact.history_temporal_layout = args.history_temporal_layout
    exact.pyramid_num_inference_steps_list = list(args.pyramid_num_inference_steps_list)
    exact.attention_backend = str(args.attention_backend)
    exact.is_amplify_first_chunk = False
    exact.seed = int(args.seed)
    exact.lora_rank = int(args.lora_rank)
    exact.lora_alpha = int(args.lora_alpha)
    exact.lora_dropout = float(args.lora_dropout)
    exact.lora_target_modules = args.lora_target_modules
    exact.lora_adapter_name = args.lora_adapter_name
    exact.flow_matching_mode = "train_exact"
    exact.flow_matching_stage_sampling = args.flow_matching_stage_sampling
    exact.flow_matching_stage_id = int(args.flow_matching_stage_id)
    exact.flow_matching_train_exact_timestep_sampling = str(args.flow_matching_train_exact_timestep_sampling)
    exact.flow_matching_use_dynamic_shifting = "off"
    exact.weighting_scheme = "none"
    exact.iters = int(args.max_steps)
    exact.lr = float(args.lr)
    exact.lr_schedule = args.lr_schedule
    exact.lr_schedule_final_ratio = float(args.lr_schedule_final_ratio)
    exact.gradient_checkpointing = bool(args.gradient_checkpointing)
    exact.overwrite = bool(args.overwrite)

    opt.validate_args(exact)
    return exact


def main():
    args = parse_args()
    exact_args = build_exact_args(args)

    out_dir = Path(args.output_dir)
    loss_path = out_dir / "train_loss.json"
    if loss_path.exists() and not args.overwrite:
        raise FileExistsError(f"{loss_path} exists. Use --overwrite to run again.")
    out_dir.mkdir(parents=True, exist_ok=True)
    tb_writer, tb_log_dir = create_tensorboard_writer(args, out_dir)

    opt.seed_global_rng(args.seed)
    device = torch.device("cuda")

    df = pd.read_csv(args.prompt_csv).head(args.limit)
    df, training_meta = normalize_online_training_dataframe(df, exact_args)
    if df.empty:
        raise ValueError(f"No training rows loaded from {args.prompt_csv}")
    exact_args.online_warp_cache = build_online_warp_training_cache(df, exact_args, device)
    skipped_rows = []

    pipe = opt.load_pipeline(exact_args, device)
    mean, std = opt.latent_stats(pipe, device)

    config = {
        "train_args": vars(args),
        "exact_args": {
            key: value
            for key, value in vars(exact_args).items()
            if isinstance(value, (str, int, float, bool, list, tuple, type(None)))
        },
        "rows": df.to_dict(orient="records"),
        "skipped_rows": skipped_rows,
        "training_data": training_meta,
        "prompt_cache_dir": str(args.prompt_cache_dir) if args.prompt_cache_dir else "",
        "tensorboard_log_dir": str(tb_log_dir) if tb_log_dir else "",
        "loss": "flow_matching_train_exact",
    }
    write_json(out_dir / "train_config.json", config)
    if tb_writer is not None:
        tb_writer.add_text("config/train_args", _json_text(vars(args)), 0)
        tb_writer.add_text("config/exact_args", _json_text(config["exact_args"]), 0)
        tb_writer.add_text("config/training_data", _json_text(training_meta), 0)

    items = LazyPreparedItems(
        pipe,
        df,
        exact_args,
        device,
        mean,
        std,
        args.prompt_cache_dir,
    )
    print(
        json.dumps(
            {
                "event": "prepared_items_ready",
                "num_items": len(items),
                "prompt_cache_dir": str(args.prompt_cache_dir) if args.prompt_cache_dir else "",
            }
        ),
        flush=True,
    )

    opt.seed_global_rng(args.seed)
    adapter_name, lora_params, lora_stats = opt.setup_visible_lora(pipe.transformer, exact_args, "shared")
    print(json.dumps(lora_stats), flush=True)

    trainable_params = list(lora_params)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    losses = []
    start_time = time.perf_counter()
    index_iter = next_index_generator(len(items), args.max_steps, args.shuffle, args.seed)
    for step in tqdm(range(args.max_steps), desc="train shared lora"):
        item_idx = next(index_iter)
        item = items.get(item_idx)
        current_lr = current_train_lr(step, args.max_steps, args, exact_args)

        opt.set_optimizer_lr(optimizer, current_lr)
        optimizer.zero_grad(set_to_none=True)
        loss, stats, _ = opt.flow_matching_loss(
            pipe,
            item["prompt_embeds"],
            item["target_latents"],
            item["histories"],
            exact_args,
            device,
        )
        loss.backward()
        grad_norm = None
        if float(args.max_grad_norm) > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(args.max_grad_norm))
        optimizer.step()
        pipe.transformer.set_adapter(adapter_name)

        record = {
            "step": int(step),
            "seq": item["seq"],
            "loss": float(loss.detach().cpu()),
            "lr": current_lr,
            "lora_rank": int(args.lora_rank),
            "lora_alpha": int(args.lora_alpha),
            "lora_target_modules": opt.lora_target_modules(exact_args),
            "optimizer": "adamw",
            "adamw_weight_decay": 0.01,
            "warmup_steps": int(args.warmup_steps),
            "max_grad_norm": float(args.max_grad_norm),
            "grad_norm": scalar(grad_norm) if grad_norm is not None else None,
            "elapsed_s": time.perf_counter() - start_time,
        }
        for key, value in stats.items():
            record[key] = scalar(value)
        losses.append(record)
        tensorboard_log_record(tb_writer, record, step)

        do_log = args.log_every > 0 and ((step + 1) % args.log_every == 0 or step == 0)
        do_save = args.save_every > 0 and (step + 1) % args.save_every == 0
        if do_log:
            print(json.dumps(record), flush=True)
            write_json(loss_path, losses)
            if tb_writer is not None:
                tb_writer.flush()
        if do_save:
            save_lora(pipe, out_dir, adapter_name, f"visible_lora_state_step{step + 1:04d}.pt")

        del loss, stats, item
        if grad_norm is not None:
            del grad_norm
        release_cuda_cache()

    save_lora(pipe, out_dir, adapter_name, "visible_lora_state.pt")
    write_json(loss_path, losses)
    if tb_writer is not None:
        tb_writer.add_text("summary/prompt_cache_status", _json_text(items.prompt_cache_status_counts), args.max_steps)
        tb_writer.flush()
        tb_writer.close()
    print(
        json.dumps(
            {
                "event": "prompt_cache_summary",
                "prompt_cache_dir": str(args.prompt_cache_dir) if args.prompt_cache_dir else "",
                "statuses": items.prompt_cache_status_counts,
            }
        ),
        flush=True,
    )
    print(
        json.dumps(
            {
                "event": "train_done",
                "output_dir": str(out_dir),
                "steps": int(args.max_steps),
                "num_items": len(items),
                "time_total_s": time.perf_counter() - start_time,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
