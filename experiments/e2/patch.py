from __future__ import annotations

import inspect
import types
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from tokenfovea.integrations.qwen.common import (
    attention,
    mrope_sections,
    qwen2_effective_rope,
    rotate_full_key,
    rotate_partial_key,
    validate_cache_result,
)  # type: ignore[import-untyped]

from .session import E2Session


@dataclass(slots=True)
class E2PatchHandle:
    session: E2Session
    hooks: list[Any]
    patched_modules: list[tuple[torch.nn.Module, Any]]

    def remove(self) -> None:
        for hook in self.hooks:
            hook.remove()
        for module, original in self.patched_modules:
            module.forward = original


def install_e2(model: torch.nn.Module, session: E2Session) -> E2PatchHandle:
    model_any: Any = model
    model_type = str(getattr(model_any.config, "model_type", ""))
    if model_type not in {"qwen2_5_vl", "qwen3_5"}:
        raise ValueError(f"unsupported E2 model type: {model_type}")
    if session.enabled and getattr(model_any.config, "_attn_implementation", None) != "sdpa":
        raise ValueError("E2 pooled conditions require attn_implementation='sdpa'")
    language_model: Any = model_any.model.language_model
    image_token_id = int(model_any.config.image_token_id)
    spatial_merge_size = int(model_any.config.vision_config.spatial_merge_size)

    def position_encoder(reference: torch.Tensor, position_ids: torch.Tensor):
        rotary = language_model.rotary_emb
        inv_freq = getattr(rotary, "inv_freq", None)
        device = reference.device if inv_freq is None or inv_freq.device.type == "meta" else inv_freq.device
        cos, sin = rotary(reference.to(device), position_ids.to(device))
        cos, sin = cos.to(reference.device), sin.to(reference.device)
        if model_type == "qwen2_5_vl":
            return qwen2_effective_rope(cos, sin, mrope_sections(language_model.config))
        return cos, sin

    layer_types = getattr(language_model.config, "layer_types", None) or ["full_attention"] * len(
        language_model.layers
    )
    layers: list[int] = []
    patched: list[tuple[torch.nn.Module, Any]] = []
    for index, layer in enumerate(language_model.layers):
        if layer_types[index] != "full_attention":
            continue
        module = layer.self_attn
        original = module.forward
        replacement = _qwen25_forward(original, session) if model_type == "qwen2_5_vl" else _qwen35_forward(original, session)
        module.forward = types.MethodType(replacement, module)
        layers.append(index)
        patched.append((module, original))
    session.attach(layers, position_encoder, model_type)

    model_signature = inspect.signature(model.forward)
    language_signature = inspect.signature(language_model.forward)

    def model_hook(_module, args, kwargs):
        arguments = _arguments(model_signature, args, kwargs)
        input_ids = arguments.get("input_ids")
        past = arguments.get("past_key_values")
        if past is None:
            past = arguments.get("past_key_value")
        new_prompt = input_ids is not None and (past is None or int(past.get_seq_length()) == 0)
        if new_prompt and session.enabled:
            grid = arguments.get("image_grid_thw")
            if grid is not None:
                if not session.pending_sample_id:
                    raise RuntimeError("call E2Session.begin_sample() before model.generate()")
                session.configure_prompt(
                    session.pending_sample_id, input_ids, grid, image_token_id, spatial_merge_size
                )
        session.start_forward()

    def model_post_hook(_module, _args, _kwargs, output):
        session.finish_forward()
        return output

    def language_hook(_module, args, kwargs):
        session.observe_position_ids(_arguments(language_signature, args, kwargs).get("position_ids"))

    hooks = [
        model.register_forward_pre_hook(model_hook, with_kwargs=True),
        model.register_forward_hook(model_post_hook, with_kwargs=True),
        language_model.register_forward_pre_hook(language_hook, with_kwargs=True),
    ]
    return E2PatchHandle(session, hooks, patched)


def _qwen25_forward(original, session: E2Session):
    def forward(
        module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ):
        if not session.enabled or not session.configured or position_embeddings is None:
            return original(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        batch, length, _ = hidden_states.shape
        query = module.q_proj(hidden_states).view(batch, length, -1, module.head_dim).transpose(1, 2)
        raw_key = module.k_proj(hidden_states).view(batch, length, -1, module.head_dim).transpose(1, 2)
        value = module.v_proj(hidden_states).view(batch, length, -1, module.head_dim).transpose(1, 2)
        cos, sin = qwen2_effective_rope(*position_embeddings, mrope_sections(module.config))
        rotated_query = rotate_full_key(query, cos, sin)
        rotated_key = rotate_full_key(raw_key, cos, sin)
        prefill = session.is_prefill_layer(module.layer_idx)
        if prefill:
            projector = None
            if session.condition.pooling == "hidden":
                def projector(nodes):
                    count = nodes.shape[1]
                    keys = module.k_proj(nodes).view(batch, count, -1, module.head_dim).transpose(1, 2)
                    values = module.v_proj(nodes).view(batch, count, -1, module.head_dim).transpose(1, 2)
                    return keys, values
            session.capture_layer(module.layer_idx, raw_key, value, rotated_key, hidden_states, projector)
            if session.condition.front_mode == "random_perstep":
                return original(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
        full_key, full_value = (
            past_key_values.update(rotated_key, value, module.layer_idx)
            if past_key_values is not None else (rotated_key, value)
        )
        if past_key_values is not None:
            validate_cache_result(past_key_values, full_key, module.layer_idx)
        output, weights = attention(
            rotated_query, full_key, full_value, module.num_key_value_groups,
            module.scaling, output_attentions, attention_mask,
        )
        if prefill:
            compact = session.prefill_compact(module.layer_idx, full_key, full_value, raw_key)
            if compact is not None:
                keys, values, query_index, mask = compact
                replacement = _compact_attention(
                    rotated_query.index_select(-2, query_index), keys, values,
                    module.num_key_value_groups, module.scaling, mask,
                )
                output.index_copy_(1, query_index, replacement)
        else:
            keys, values, mask = session.decode_compact(module.layer_idx, full_key, full_value, raw_key)
            output = _compact_attention(
                rotated_query, keys, values, module.num_key_value_groups,
                module.scaling, mask,
            )
            session.finish_layer(module.layer_idx)
        return module.o_proj(output.reshape(batch, length, -1)), weights
    return forward


def _qwen35_forward(original, session: E2Session):
    def forward(
        module,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        **kwargs,
    ):
        if not session.enabled or not session.configured:
            return original(
                hidden_states, position_embeddings=position_embeddings,
                attention_mask=attention_mask, past_key_values=past_key_values, **kwargs,
            )
        shape = hidden_states.shape[:-1]
        head_shape = (*shape, -1, module.head_dim)
        query_gate = module.q_proj(hidden_states).view(*shape, -1, module.head_dim * 2)
        query, gate = torch.chunk(query_gate, 2, dim=-1)
        gate = gate.reshape(*shape, -1)
        query = module.q_norm(query.view(head_shape)).transpose(1, 2)
        raw_key = module.k_norm(module.k_proj(hidden_states).view(head_shape)).transpose(1, 2)
        value = module.v_proj(hidden_states).view(head_shape).transpose(1, 2)
        cos, sin = position_embeddings
        rotated_query = rotate_partial_key(query, cos, sin)
        rotated_key = rotate_partial_key(raw_key, cos, sin)
        prefill = session.is_prefill_layer(module.layer_idx)
        if prefill:
            projector = None
            if session.condition.pooling == "hidden":
                def projector(nodes):
                    count = nodes.shape[1]
                    keys = module.k_proj(nodes).view(nodes.shape[0], count, -1, module.head_dim)
                    keys = module.k_norm(keys).transpose(1, 2)
                    values = module.v_proj(nodes).view(nodes.shape[0], count, -1, module.head_dim).transpose(1, 2)
                    return keys, values
            session.capture_layer(module.layer_idx, raw_key, value, rotated_key, hidden_states, projector)
            if session.condition.front_mode == "random_perstep":
                return original(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    **kwargs,
                )
        full_key, full_value = (
            past_key_values.update(rotated_key, value, module.layer_idx)
            if past_key_values is not None else (rotated_key, value)
        )
        if past_key_values is not None:
            validate_cache_result(past_key_values, full_key, module.layer_idx)
        output_attentions = bool(kwargs.get("output_attentions", False))
        output, weights = attention(
            rotated_query, full_key, full_value, module.num_key_value_groups,
            module.scaling, output_attentions, attention_mask,
        )
        if prefill:
            compact = session.prefill_compact(module.layer_idx, full_key, full_value, raw_key)
            if compact is not None:
                keys, values, query_index, mask = compact
                replacement = _compact_attention(
                    rotated_query.index_select(-2, query_index), keys, values,
                    module.num_key_value_groups, module.scaling, mask,
                )
                output.index_copy_(1, query_index, replacement)
        else:
            keys, values, mask = session.decode_compact(module.layer_idx, full_key, full_value, raw_key)
            output = _compact_attention(
                rotated_query, keys, values, module.num_key_value_groups,
                module.scaling, mask,
            )
            session.finish_layer(module.layer_idx)
        output = output.reshape(*shape, -1) * torch.sigmoid(gate)
        return module.o_proj(output), weights
    return forward


def _compact_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    groups: int,
    scaling: float,
    mask: torch.Tensor,
) -> torch.Tensor:
    """SDPA using E2's original-position causal mask without index-based remasking."""

    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=mask.to(query.device),
        dropout_p=0.0,
        is_causal=False,
        scale=scaling,
        enable_gqa=groups > 1,
    )
    return output.transpose(1, 2).contiguous()


def _arguments(signature: inspect.Signature, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    arguments = dict(kwargs)
    bound = signature.bind_partial(*args, **kwargs)
    for name, value in bound.arguments.items():
        if signature.parameters[name].kind == inspect.Parameter.VAR_KEYWORD:
            arguments.update(value)
        else:
            arguments[name] = value
    return arguments
