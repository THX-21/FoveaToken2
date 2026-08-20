from __future__ import annotations

from typing import Any

import torch


def generate_with_prefill_boundary(
    model: Any,
    inputs: dict[str, Any],
    generation_kwargs: dict[str, Any],
) -> Any:
    """Prefill through the penultimate prompt token, then generate from the final token."""
    if "past_key_values" in generation_kwargs:
        raise ValueError("prefill-boundary generation manages past_key_values internally")
    prefix_output = model(
        **prompt_prefix_inputs(inputs),
        use_cache=True,
        return_dict=True,
    )
    past_key_values = getattr(prefix_output, "past_key_values", None)
    if past_key_values is None:
        raise RuntimeError("prefill-boundary forward did not return a KV cache")
    return model.generate(
        **inputs,
        past_key_values=past_key_values,
        **generation_kwargs,
    )


def prompt_prefix_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Remove only the final prompt token from token-aligned model inputs."""
    input_ids = inputs.get("input_ids")
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise ValueError("prefill-boundary generation requires rank-2 input_ids")
    prompt_length = input_ids.shape[-1]
    if prompt_length < 2:
        raise ValueError("prefill-boundary generation requires at least two prompt tokens")
    prefix = dict(inputs)
    prefix["input_ids"] = input_ids[..., :-1]
    for name in ("attention_mask", "token_type_ids", "position_ids", "cache_position"):
        value = inputs.get(name)
        if isinstance(value, torch.Tensor) and value.shape[-1] == prompt_length:
            prefix[name] = value[..., :-1]
    return prefix
