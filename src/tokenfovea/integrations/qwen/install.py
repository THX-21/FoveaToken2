from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any

import torch

from ...session import FoveaSession
from .common import mrope_sections, qwen2_effective_rope, rotate_full_key, rotate_partial_key
from .qwen25_vl import make_forward as make_qwen25_vl_forward
from .qwen35 import make_forward as make_qwen35_forward


@dataclass(slots=True)
class PatchHandle:
    session: FoveaSession
    hooks: list[Any]
    patched_modules: list[tuple[torch.nn.Module, Any]]

    def remove(self) -> None:
        for hook in self.hooks:
            hook.remove()
        for module, original in self.patched_modules:
            module.forward = original


def install_tokenfovea(model: torch.nn.Module, session: FoveaSession) -> PatchHandle:
    """Patch Qwen full-attention layers while leaving prompt prefill unchanged."""
    model_any: Any = model
    model_type = getattr(model_any.config, "model_type", "")
    if model_type not in {"qwen2_5_vl", "qwen3_5"}:
        raise ValueError(f"unsupported model_type: {model_type}")
    language_model: Any = model_any.model.language_model
    image_token_id = int(model_any.config.image_token_id)
    spatial_merge_size = int(model_any.config.vision_config.spatial_merge_size)

    def position_encoder(reference: torch.Tensor, position_ids: torch.Tensor):
        rotary = language_model.rotary_emb
        inv_freq = getattr(rotary, "inv_freq", None)
        rotary_device = reference.device if inv_freq is None or inv_freq.device.type == "meta" else inv_freq.device
        cos, sin = rotary(reference.to(rotary_device), position_ids.to(rotary_device))
        cos, sin = cos.to(reference.device), sin.to(reference.device)
        if model_type == "qwen2_5_vl":
            return qwen2_effective_rope(cos, sin, mrope_sections(language_model.config))
        return cos, sin

    rotate_key = rotate_full_key if model_type == "qwen2_5_vl" else rotate_partial_key
    routed_layers = []
    patched_modules = []
    layer_types = getattr(language_model.config, "layer_types", None) or ["full_attention"] * len(
        language_model.layers
    )
    for layer_index, layer in enumerate(language_model.layers):
        if layer_types[layer_index] != "full_attention":
            continue
        module = layer.self_attn
        original = module.forward
        replacement = (
            make_qwen25_vl_forward(original, session)
            if model_type == "qwen2_5_vl"
            else make_qwen35_forward(original, session)
        )
        module.forward = types.MethodType(replacement, module)
        routed_layers.append(layer_index)
        patched_modules.append((module, original))
    session.attach(routed_layers, position_encoder, rotate_key)

    def model_pre_hook(_module, args, kwargs):
        input_ids = kwargs.get("input_ids")
        inputs_embeds = kwargs.get("inputs_embeds")
        past_key_values = kwargs.get("past_key_values")
        image_grid = kwargs.get("image_grid_thw")
        video_grid = kwargs.get("video_grid_thw")
        new_prompt = (input_ids is not None or inputs_embeds is not None) and (
            past_key_values is None or past_key_values.get_seq_length() == 0
        )
        if not new_prompt:
            return
        session.reset_prompt()
        if not session.enabled:
            return
        if video_grid is not None:
            raise ValueError("TokenFovea currently supports images only")
        if image_grid is not None:
            if input_ids is None:
                raise ValueError("TokenFovea image prompts require input_ids")
            session.configure_prompt(input_ids, image_grid, image_token_id, spatial_merge_size)

    def language_pre_hook(_module, args, kwargs):
        session.observe_position_ids(kwargs.get("position_ids"))

    hooks = [
        model.register_forward_pre_hook(model_pre_hook, with_kwargs=True),
        language_model.register_forward_pre_hook(language_pre_hook, with_kwargs=True),
    ]
    return PatchHandle(session=session, hooks=hooks, patched_modules=patched_modules)
