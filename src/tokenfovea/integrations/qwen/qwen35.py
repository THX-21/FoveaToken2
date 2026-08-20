from __future__ import annotations

import torch

from ...session import FoveaSession
from .common import (
    attention,
    compact_attention,
    rotate_partial_key,
    validate_cache_result,
    visual_attention_signal,
)


def make_forward(original, session: FoveaSession):
    def forward(
        module,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        **kwargs,
    ):
        if not session.is_configured or not session.enabled:
            return original(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )

        if session.is_prefill_layer(module.layer_idx):
            if not bool(kwargs.get("use_cache", True)):
                raise ValueError("TokenFovea requires use_cache=True")
            collect_signal = session.needs_prefill_signal(module.layer_idx)
            result = original(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, module.head_dim)
            query_and_gate = module.q_proj(hidden_states).view(
                *input_shape, -1, module.head_dim * 2
            )
            query, gate = torch.chunk(query_and_gate, 2, dim=-1)
            gate = gate.reshape(*input_shape, -1)
            query = module.q_norm(query.view(hidden_shape)).transpose(1, 2)
            raw_key = module.k_norm(
                module.k_proj(hidden_states).view(hidden_shape)
            ).transpose(1, 2)
            value = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            rotated_query = rotate_partial_key(query, cos, sin)
            rotated_key = rotate_partial_key(raw_key, cos, sin)
            projector = None
            if session.config.pooling_mode == "hidden":

                def projector(nodes):
                    node_count = nodes.shape[1]
                    keys = module.k_proj(nodes).view(hidden_states.shape[0], node_count, -1, module.head_dim)
                    keys = module.k_norm(keys).transpose(1, 2)
                    node_values = module.v_proj(nodes).view(hidden_states.shape[0], node_count, -1, module.head_dim)
                    return keys, node_values.transpose(1, 2)

            session.capture_prefill_layer(
                module.layer_idx,
                raw_key,
                value,
                rotated_key,
                None,
                hidden_states,
                projector,
            )
            if session.native_capture_scale is not None:
                return result
            keys, values, query_index, mask, active_ids = session.compose_prefill(
                module.layer_idx,
                rotated_key,
                value,
                raw_key,
            )
            compact = compact_attention(
                rotated_query.index_select(-2, query_index),
                keys,
                values,
                module.num_key_value_groups,
                module.scaling,
                mask,
            )
            compact = compact.reshape(hidden_states.shape[0], query_index.numel(), -1)
            replacement = module.o_proj(
                compact * torch.sigmoid(gate.index_select(1, query_index))
            )
            output = result[0].clone()
            output.index_copy_(1, query_index, replacement)
            visual_signal = (
                visual_attention_signal(
                    rotated_query[..., -1:, :],
                    keys[..., : active_ids.numel(), :],
                    module.num_key_value_groups,
                    module.scaling,
                )
                if collect_signal
                else None
            )
            session.record_prefill_layer(
                module.layer_idx, visual_signal, active_ids
            )
            return (output, *result[1:])

        if past_key_values is None:
            raise RuntimeError("TokenFovea decode requires a populated KV cache")
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, module.head_dim)
        query_and_gate = module.q_proj(hidden_states).view(
            *input_shape, -1, module.head_dim * 2
        )
        query, gate = torch.chunk(query_and_gate, 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)
        query = module.q_norm(query.view(hidden_shape)).transpose(1, 2)
        raw_key = module.k_norm(
            module.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        value = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        rotated_query = rotate_partial_key(query, cos, sin)
        rotated_key = rotate_partial_key(raw_key, cos, sin)
        session.validate_decode_batch(hidden_states.shape[0])
        full_key, full_value = past_key_values.update(rotated_key, value, module.layer_idx)
        validate_cache_result(past_key_values, full_key, module.layer_idx)
        key, composed_value, active_ids = session.compose(module.layer_idx, full_key, full_value, raw_key)
        composed_mask = session.compose_attention_mask(attention_mask, active_ids, full_key.shape[-2])
        visual_signal = (
            visual_attention_signal(
                rotated_query,
                key[..., : active_ids.numel(), :],
                module.num_key_value_groups,
                module.scaling,
            )
            if session.needs_signal(module.layer_idx)
            else None
        )
        output_attentions = bool(kwargs.get("output_attentions", False))
        output, weights = attention(
            rotated_query,
            key,
            composed_value,
            module.num_key_value_groups,
            module.scaling,
            output_attentions,
            composed_mask,
        )
        session.record_decode_layer(module.layer_idx, visual_signal)
        output = output.reshape(*input_shape, -1) * torch.sigmoid(gate)
        return module.o_proj(output), weights

    return forward
