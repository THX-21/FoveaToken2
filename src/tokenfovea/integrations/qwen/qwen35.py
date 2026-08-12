from __future__ import annotations

import torch

from ...session import FoveaSession
from .common import attention, rotate_partial_key, visual_attention_signal


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

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, module.head_dim)
        query_and_gate = module.q_proj(hidden_states).view(*input_shape, -1, module.head_dim * 2)
        query, gate = torch.chunk(query_and_gate, 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)
        query = module.q_norm(query.view(hidden_shape)).transpose(1, 2)
        raw_key = module.k_norm(module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        rotated_query = rotate_partial_key(query, cos, sin)
        rotated_key = rotate_partial_key(raw_key, cos, sin)

        if session.is_prefill_layer(module.layer_idx):
            if not bool(kwargs.get("use_cache", True)):
                raise ValueError("TokenFovea requires use_cache=True")
            visual_signal = None
            if session.needs_signal(module.layer_idx):
                visual_index = session.visual_index(rotated_key.device)
                visual_signal = visual_attention_signal(
                    rotated_query[..., -1:, :],
                    rotated_key.index_select(-2, visual_index),
                    module.num_key_value_groups,
                    module.scaling,
                )
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
                visual_signal,
                hidden_states,
                projector,
            )
            return original(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )

        if past_key_values is None:
            raise RuntimeError("TokenFovea decode requires a populated KV cache")
        session.validate_decode_batch(hidden_states.shape[0])
        full_key, full_value = past_key_values.update(rotated_key, value, module.layer_idx)
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
