#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
from pathlib import Path

import torch

from warp_as_history.training import core as opt


def detach_tree(value):
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, dict):
        return {key: detach_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(detach_tree(item) for item in value)
    return value


def scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().float().mean().cpu())
    if isinstance(value, (int, float)):
        return float(value)
    return value


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def release_cuda_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def next_index_generator(num_items, max_steps, shuffle, seed):
    if num_items <= 0:
        raise ValueError("No training items.")
    generator = torch.Generator().manual_seed(int(seed))
    order = []
    cursor = 0
    for step in range(int(max_steps)):
        if not shuffle:
            yield step % num_items
            continue
        if cursor >= len(order):
            order = torch.randperm(num_items, generator=generator).tolist()
            cursor = 0
        idx = order[cursor]
        cursor += 1
        yield idx


def save_lora(pipe, out_dir, adapter_name, filename):
    opt.save_visible_lora_state(pipe.transformer, out_dir, adapter_name, filename)


def current_train_lr(step, total_steps, args, exact_args):
    lr = float(args.lr) * opt.lr_schedule_multiplier(step, total_steps, exact_args)
    if int(args.warmup_steps) > 0:
        lr *= min(1.0, float(step + 1) / float(args.warmup_steps))
    return lr
