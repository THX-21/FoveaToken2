from __future__ import annotations

import torch

from ...session import FoveaSession
from .common import (
    attention,
    compact_attention,
    mrope_sections,
    qwen2_effective_rope,
    rotate_full_key,
    validate_cache_result,
    visual_attention_signal,
)


def make_forward(original, session: FoveaSession):
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
        if not session.is_configured or not session.enabled or position_embeddings is None:
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

        if session.is_prefill_layer(module.layer_idx):
            if not use_cache:
                raise ValueError("TokenFovea requires use_cache=True")
            collect_signal = session.needs_prefill_signal(module.layer_idx)
            result = original(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            batch, query_length, _ = hidden_states.shape
            query = module.q_proj(hidden_states).view(
                batch, query_length, -1, module.head_dim
            ).transpose(1, 2)
            raw_key = module.k_proj(hidden_states).view(
                batch, query_length, -1, module.head_dim
            ).transpose(1, 2)
            value = module.v_proj(hidden_states).view(
                batch, query_length, -1, module.head_dim
            ).transpose(1, 2)
            cos, sin = qwen2_effective_rope(
                *position_embeddings, mrope_sections(module.config)
            )
            rotated_query = rotate_full_key(query, cos, sin)
            rotated_key = rotate_full_key(raw_key, cos, sin)
            projector = None
            if session.config.pooling_mode == "hidden":

                def projector(nodes):
                    node_count = nodes.shape[1]
                    keys = module.k_proj(nodes).view(batch, node_count, -1, module.head_dim).transpose(1, 2)
                    node_values = module.v_proj(nodes).view(batch, node_count, -1, module.head_dim).transpose(1, 2)
                    return keys, node_values

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
            replacement = module.o_proj(
                compact.reshape(batch, query_index.numel(), -1)
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
        batch, query_length, _ = hidden_states.shape
        query = module.q_proj(hidden_states).view(
            batch, query_length, -1, module.head_dim
        ).transpose(1, 2)
        raw_key = module.k_proj(hidden_states).view(
            batch, query_length, -1, module.head_dim
        ).transpose(1, 2)
        value = module.v_proj(hidden_states).view(
            batch, query_length, -1, module.head_dim
        ).transpose(1, 2)
        cos, sin = qwen2_effective_rope(
            *position_embeddings, mrope_sections(module.config)
        )
        rotated_query = rotate_full_key(query, cos, sin)
        rotated_key = rotate_full_key(raw_key, cos, sin)
        session.validate_decode_batch(batch)
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
        return module.o_proj(output.reshape(batch, query_length, -1)), weights

    return forward
