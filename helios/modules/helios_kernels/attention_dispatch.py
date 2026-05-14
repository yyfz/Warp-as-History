import os

import torch


try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    print("Flash Attn 2 is installed!")
except (ImportError, RuntimeError, OSError):
    print("Flash Attn 2 is not installed!")
    flash_attn_varlen_func = None
    flash_attn_func = None


try:
    # raise NotImplementedError
    from sageattention import sageattn, sageattn_varlen

    print("Sage Attn is installed!")
except ImportError:
    print("Sage Attn is not installed!")
    sageattn_varlen = None
    sageattn = None

try:
    # raise NotImplementedError
    if os.environ.get("XFORMERS_DISABLED") is not None:
        raise ImportError
    from xformers.ops import memory_efficient_attention as xformers_attn_func

    print("Xformers is installed!")
except (ImportError, RuntimeError, OSError):
    print("Xformers is not installed!")
    xformers_attn_func = None


def create_navit_attention_masks(
    batch_size: int,
    original_context_length_list: list,
    history_context_length: int,
    encoder_hidden_states_seq_len: int,
    device: torch.device,
    restrict_self_attn: bool = False,
    guidance_cross_attn: bool = False,
):
    # For navit_hidden_attention_mask
    if restrict_self_attn:
        cu_seqlens_q = [0]
        for _ in range(batch_size):
            for length in original_context_length_list:
                cu_seqlens_q.append(cu_seqlens_q[-1] + length)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, device=device, dtype=torch.int32)
        max_seqlen_q = max(original_context_length_list)

        cu_seqlens_kv = [0]
        for _ in range(batch_size):
            for length in original_context_length_list:
                cu_seqlens_kv.append(cu_seqlens_kv[-1] + length + history_context_length)
        cu_seqlens_kv = torch.tensor(cu_seqlens_kv, device=device, dtype=torch.int32)
        max_seqlen_kv = max(original_context_length_list) + history_context_length
    else:
        cu_seqlens_kv = [0]
        for _ in range(batch_size):
            for length in original_context_length_list:
                cu_seqlens_kv.append(cu_seqlens_kv[-1] + length + history_context_length)
        cu_seqlens_kv = torch.tensor(cu_seqlens_kv, device=device, dtype=torch.int32)
        max_seqlen_kv = max(original_context_length_list) + history_context_length
        cu_seqlens_q = cu_seqlens_kv
        max_seqlen_q = max_seqlen_kv
    navit_hidden_attention_mask = cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv

    # For navit_history_hidden_attention_mask
    navit_history_hidden_attention_mask = None
    if restrict_self_attn:
        cu_seqlens_kv = [0]
        for _ in range(batch_size):
            for length in original_context_length_list:
                cu_seqlens_kv.append(cu_seqlens_kv[-1] + history_context_length)
        cu_seqlens_kv = torch.tensor(cu_seqlens_kv, device=device, dtype=torch.int32)
        max_seqlen_kv = history_context_length
        cu_seqlens_q = cu_seqlens_kv
        max_seqlen_q = max_seqlen_kv
        navit_history_hidden_attention_mask = cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv

    # For navit_encoder_attention_mask
    if guidance_cross_attn:
        cross_cu_seqlens_q = [0]
        for _ in range(batch_size):
            for length in original_context_length_list:
                cross_cu_seqlens_q.append(cross_cu_seqlens_q[-1] + length)
        cross_cu_seqlens_q = torch.tensor(cross_cu_seqlens_q, device=device, dtype=torch.int32)
        cross_max_seqlen_q = max(original_context_length_list)
    else:
        cross_cu_seqlens_q = [0]
        for _ in range(batch_size):
            for length in original_context_length_list:
                cross_cu_seqlens_q.append(cross_cu_seqlens_q[-1] + length + history_context_length)
        cross_cu_seqlens_q = torch.tensor(cross_cu_seqlens_q, device=device, dtype=torch.int32)
        cross_cu_seqlens_q[0] = 0
        cross_max_seqlen_q = max(original_context_length_list) + history_context_length

    cu_seqlens_kv = [0]
    for _ in range(batch_size):
        for length in original_context_length_list:
            cu_seqlens_kv.append(cu_seqlens_kv[-1] + encoder_hidden_states_seq_len)
    cu_seqlens_kv = torch.tensor(cu_seqlens_kv, device=device, dtype=torch.int32)
    max_seqlen_kv = encoder_hidden_states_seq_len
    navit_encoder_attention_mask = cross_cu_seqlens_q, cu_seqlens_kv, cross_max_seqlen_q, max_seqlen_kv

    return navit_hidden_attention_mask, navit_encoder_attention_mask, navit_history_hidden_attention_mask


@torch.compiler.disable
def _flash_attn_wrapper(q, k, v):
    return flash_attn_func(q, k, v)


@torch.compiler.disable
def _flash_attn_varlen_wrapper(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv):
    return flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)


def attn_varlen_func(q, k, v, attention_mask=None):
    if attention_mask is None:
        if flash_attn_func is not None:
            x = _flash_attn_wrapper(q, k, v)
            return x

        if sageattn is not None:
            x = sageattn(q, k, v, tensor_layout="NHD")
            return x

        if xformers_attn_func is not None:
            x = xformers_attn_func(q, k, v)
            return x

        x = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)
        return x

    B, L, H, C = q.shape

    q = q.flatten(0, 1)
    k = k.flatten(0, 1)
    v = v.flatten(0, 1)

    cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv = attention_mask
    if flash_attn_varlen_func is not None:
        x = _flash_attn_varlen_wrapper(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)
    elif sageattn_varlen is not None:
        x = sageattn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)
    else:
        outputs = []
        for start_q, end_q, start_kv, end_kv in zip(
            cu_seqlens_q[:-1].tolist(),
            cu_seqlens_q[1:].tolist(),
            cu_seqlens_kv[:-1].tolist(),
            cu_seqlens_kv[1:].tolist(),
        ):
            q_i = q[int(start_q) : int(end_q)].unsqueeze(0)
            k_i = k[int(start_kv) : int(end_kv)].unsqueeze(0)
            v_i = v[int(start_kv) : int(end_kv)].unsqueeze(0)
            out_i = torch.nn.functional.scaled_dot_product_attention(
                q_i.transpose(1, 2),
                k_i.transpose(1, 2),
                v_i.transpose(1, 2),
            ).transpose(1, 2)[0]
            outputs.append(out_i)
        x = torch.cat(outputs, dim=0)

    x = x.unflatten(0, (B, L))

    return x
