from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def qwen2_effective_rope(
    cos: torch.Tensor,
    sin: torch.Tensor,
    sections: list[int] | tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    sizes = [int(value) * 2 for value in sections]
    cos_eff = torch.cat([part[i % 3] for i, part in enumerate(cos.split(sizes, dim=-1))], dim=-1)
    sin_eff = torch.cat([part[i % 3] for i, part in enumerate(sin.split(sizes, dim=-1))], dim=-1)
    return cos_eff, sin_eff


def rotate_full_key(key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return key * cos + rotate_half(key) * sin


def rotate_partial_key(key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    rotated, passthrough = key[..., :rotary_dim], key[..., rotary_dim:]
    rotated = rotated * cos + rotate_half(rotated) * sin
    return torch.cat((rotated, passthrough), dim=-1)


def repeat_kv(states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return states
    batch, kv_heads, seq_len, head_dim = states.shape
    return (
        states[:, :, None]
        .expand(batch, kv_heads, repeats, seq_len, head_dim)
        .reshape(batch, kv_heads * repeats, seq_len, head_dim)
    )


def validate_cache_result(cache: Any, keys: torch.Tensor, layer_idx: int) -> None:
    get_seq_length = getattr(cache, "get_seq_length", None)
    if not callable(get_seq_length):
        raise TypeError("TokenFovea requires a cache exposing get_seq_length()")
    logical_length = int(get_seq_length(layer_idx))
    physical_length = keys.shape[-2]
    if logical_length != physical_length:
        raise ValueError(
            "TokenFovea requires cache updates to return only populated K/V entries; "
            f"layer {layer_idx} returned {physical_length} entries for a logical length of {logical_length}"
        )


def attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    groups: int,
    scaling: float,
    output_attentions: bool,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if attention_mask is not None:
        attention_mask = attention_mask.to(query.device)
        if attention_mask.ndim == 2:
            if attention_mask.dtype == torch.bool or not attention_mask.is_floating_point():
                attention_mask = attention_mask != 0
            else:
                attention_mask = attention_mask > 0
            attention_mask = attention_mask[:, None, None, :]
        elif not (attention_mask.dtype == torch.bool or attention_mask.is_floating_point()):
            attention_mask = attention_mask != 0
        elif attention_mask.is_floating_point():
            attention_mask = attention_mask.to(query.dtype)
    query_length = query.shape[-2]
    key_length = key.shape[-2]
    use_sdpa_causal = (
        query_length > 1 and query_length == key_length and attention_mask is None and not output_attentions
    )
    if query_length > 1:
        if key_length < query_length:
            raise ValueError("self-attention key length cannot be shorter than query length")
        if not use_sdpa_causal:
            query_positions = torch.arange(query_length, device=query.device)[:, None]
            key_positions = torch.arange(key_length, device=query.device)[None, :]
            causal_mask = key_positions <= query_positions + key_length - query_length
            causal_mask = causal_mask[None, None]
            if attention_mask is None:
                attention_mask = causal_mask
            elif attention_mask.dtype == torch.bool:
                attention_mask = attention_mask & causal_mask
            else:
                attention_mask = attention_mask.masked_fill(~causal_mask, -torch.inf)
    if output_attentions:
        repeated_key = repeat_kv(key, groups)
        repeated_value = repeat_kv(value, groups)
        logits = torch.matmul(query, repeated_key.transpose(-2, -1)) * scaling
        if attention_mask is not None:
            logits = (
                logits + attention_mask
                if attention_mask.dtype != torch.bool
                else logits.masked_fill(~attention_mask, -torch.inf)
            )
        weights = torch.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
        output = torch.matmul(weights, repeated_value)
    else:
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=use_sdpa_causal,
            scale=scaling,
            enable_gqa=groups > 1,
        )
        weights = None
    return output.transpose(1, 2).contiguous(), weights


def compact_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    groups: int,
    scaling: float,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Attend to a compact KV sequence using an explicit original-position mask."""
    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask.to(query.device),
        dropout_p=0.0,
        is_causal=False,
        scale=scaling,
        enable_gqa=groups > 1,
    )
    return output.transpose(1, 2).contiguous()


def visual_attention_signal(
    query: torch.Tensor,
    visual_keys: torch.Tensor,
    groups: int,
    scaling: float,
) -> torch.Tensor:
    """Compute only the small visual distribution used by the router."""
    batch, kv_heads, _, head_dim = visual_keys.shape
    grouped_query = query.reshape(batch, kv_heads, groups, query.shape[-2], head_dim)
    logits = torch.einsum("bhgqd,bhkd->bhgqk", grouped_query, visual_keys) * scaling
    weights = torch.softmax(logits, dim=-1, dtype=torch.float32).mean(dim=(0, 3))
    return weights.reshape(kv_heads * groups, visual_keys.shape[-2])


def mrope_sections(config: Any) -> list[int]:
    parameters = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
    sections = parameters.get("mrope_section")
    if sections is None:
        raise ValueError("Qwen M-RoPE configuration does not expose mrope_section")
    return list(sections)
