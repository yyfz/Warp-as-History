from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import torch

from .utils_base import prompt_clean


PROMPT_EMBED_CACHE_FORMAT = "helios_prompt_embeds_v1"


def load_prompt_map(prompt_csv: str | Path | None) -> dict[str, str]:
    if prompt_csv is None:
        return {}

    path = Path(prompt_csv).expanduser()
    if not path.exists():
        return {}

    prompt_map: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            video_id = row.get("id")
            if not video_id:
                continue
            prompt = row.get("refined_prompt") or row.get("prompt") or ""
            prompt_map[str(video_id)] = str(prompt).strip()
    return prompt_map


def discover_cache_video_ids(cache_root: str | Path) -> list[str]:
    root = Path(cache_root).expanduser().resolve()
    video_ids: list[str] = []
    for meta_path in sorted(root.glob("*/meta.json")):
        video_ids.append(meta_path.parent.name)
    return video_ids


def format_prompt(prompt: str, prompt_prefix: str = "", prompt_suffix: str = "") -> str:
    parts = [part.strip() for part in (prompt_prefix, prompt, prompt_suffix) if part and part.strip()]
    return " ".join(parts).strip()


def build_video_prompts(
    video_ids: Iterable[str],
    prompt_map: dict[str, str],
    *,
    empty_prompt: str = "",
    prompt_prefix: str = "",
    prompt_suffix: str = "",
) -> dict[str, str]:
    fallback_prompt = format_prompt(empty_prompt, prompt_prefix=prompt_prefix, prompt_suffix=prompt_suffix)
    video_prompts: dict[str, str] = {}
    for video_id in video_ids:
        raw_prompt = prompt_map.get(video_id, empty_prompt)
        formatted_prompt = format_prompt(raw_prompt, prompt_prefix=prompt_prefix, prompt_suffix=prompt_suffix)
        video_prompts[video_id] = formatted_prompt if formatted_prompt else fallback_prompt
    return video_prompts


@torch.no_grad()
def encode_prompt_batch(
    tokenizer,
    text_encoder,
    prompts: list[str],
    *,
    max_sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    cleaned_prompts = [prompt_clean(prompt) for prompt in prompts]
    text_inputs = tokenizer(
        cleaned_prompts,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)
    seq_lens = attention_mask.gt(0).sum(dim=1).long()

    prompt_embeds = text_encoder(input_ids, attention_mask).last_hidden_state
    prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    prompt_embeds = [embed[:seq_len] for embed, seq_len in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [
            torch.cat([embed, embed.new_zeros(max_sequence_length - embed.size(0), embed.size(1))], dim=0)
            for embed in prompt_embeds
        ],
        dim=0,
    )
    return prompt_embeds, text_inputs.attention_mask.bool()


@torch.no_grad()
def build_prompt_embed_cache(
    tokenizer,
    text_encoder,
    video_prompts: dict[str, str],
    *,
    empty_prompt: str,
    prompt_prefix: str,
    prompt_suffix: str,
    max_sequence_length: int,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 16,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    fallback_prompt = format_prompt(empty_prompt, prompt_prefix=prompt_prefix, prompt_suffix=prompt_suffix)
    unique_prompts = sorted({fallback_prompt, *video_prompts.values()})
    prompt_embeds_by_text: dict[str, torch.Tensor] = {}
    prompt_attention_masks_by_text: dict[str, torch.Tensor] = {}

    for start in range(0, len(unique_prompts), batch_size):
        prompt_batch = unique_prompts[start : start + batch_size]
        prompt_embeds, prompt_attention_mask = encode_prompt_batch(
            tokenizer,
            text_encoder,
            prompt_batch,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
        for index, prompt_text in enumerate(prompt_batch):
            prompt_embeds_by_text[prompt_text] = prompt_embeds[index].detach().cpu()
            prompt_attention_masks_by_text[prompt_text] = prompt_attention_mask[index].detach().cpu()

    cache: dict[str, object] = {
        "format": PROMPT_EMBED_CACHE_FORMAT,
        "video_prompts": dict(video_prompts),
        "prompt_embeds": {
            video_id: prompt_embeds_by_text[prompt_text] for video_id, prompt_text in video_prompts.items()
        },
        "prompt_attention_masks": {
            video_id: prompt_attention_masks_by_text[prompt_text] for video_id, prompt_text in video_prompts.items()
        },
        "empty_prompt": empty_prompt,
        "empty_prompt_formatted": fallback_prompt,
        "empty_prompt_embeds": prompt_embeds_by_text[fallback_prompt],
        "empty_prompt_attention_mask": prompt_attention_masks_by_text[fallback_prompt],
        "prompt_prefix": prompt_prefix,
        "prompt_suffix": prompt_suffix,
        "max_sequence_length": int(max_sequence_length),
    }
    if metadata:
        cache.update(metadata)
    return cache


def save_prompt_embed_cache(cache: dict[str, object], path: str | Path) -> Path:
    cache_path = Path(path).expanduser().resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    return cache_path


def load_prompt_embed_cache(path: str | Path) -> dict[str, object]:
    cache_path = Path(path).expanduser().resolve()
    return torch.load(cache_path, map_location="cpu")


def prompt_embed_cache_mismatches(
    cache: dict[str, object],
    *,
    video_prompts: dict[str, str],
    max_sequence_length: int,
    prompt_prefix: str,
    prompt_suffix: str,
    empty_prompt: str,
    base_model_path: str | None = None,
) -> list[str]:
    mismatches: list[str] = []

    if cache.get("format") != PROMPT_EMBED_CACHE_FORMAT:
        mismatches.append(
            f"format={cache.get('format')!r} expected={PROMPT_EMBED_CACHE_FORMAT!r}"
        )
    if int(cache.get("max_sequence_length", -1)) != int(max_sequence_length):
        mismatches.append(
            f"max_sequence_length={cache.get('max_sequence_length')} expected={int(max_sequence_length)}"
        )
    if str(cache.get("prompt_prefix", "")) != str(prompt_prefix):
        mismatches.append(f"prompt_prefix mismatch: {cache.get('prompt_prefix')!r} vs {prompt_prefix!r}")
    if str(cache.get("prompt_suffix", "")) != str(prompt_suffix):
        mismatches.append(f"prompt_suffix mismatch: {cache.get('prompt_suffix')!r} vs {prompt_suffix!r}")
    if str(cache.get("empty_prompt", "")) != str(empty_prompt):
        mismatches.append(f"empty_prompt mismatch: {cache.get('empty_prompt')!r} vs {empty_prompt!r}")
    if base_model_path is not None and cache.get("base_model_path") not in {None, base_model_path}:
        mismatches.append(f"base_model_path mismatch: {cache.get('base_model_path')!r} vs {base_model_path!r}")

    cached_video_prompts = cache.get("video_prompts")
    if not isinstance(cached_video_prompts, dict):
        mismatches.append("video_prompts missing")
        return mismatches

    missing = sorted(set(video_prompts) - set(cached_video_prompts))
    if missing:
        mismatches.append(f"missing video ids: {missing[:5]}")
        return mismatches

    for video_id, prompt_text in video_prompts.items():
        cached_prompt = cached_video_prompts.get(video_id)
        if cached_prompt != prompt_text:
            mismatches.append(f"prompt mismatch for {video_id}: {cached_prompt!r} vs {prompt_text!r}")
            break

    if "prompt_embeds" not in cache or "empty_prompt_embeds" not in cache:
        mismatches.append("prompt embeddings missing")

    return mismatches
