import types
import unittest

import torch

from tokenfovea import FoveaConfig, FoveaSession, install_tokenfovea


class _Attention(torch.nn.Module):
    def forward(self, hidden_states, **kwargs):
        return hidden_states, None


class _DecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()


class _LanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_DecoderLayer(), _DecoderLayer()])
        self.config = types.SimpleNamespace(rope_scaling={"mrope_section": [1, 1, 1]})
        self.rotary_emb = torch.nn.Identity()

    def forward(self, **kwargs):
        return kwargs


class _Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _LanguageModel()


class _Qwen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Backbone()
        self.config = types.SimpleNamespace(
            model_type="qwen2_5_vl",
            _attn_implementation="sdpa",
            image_token_id=99,
            vision_config=types.SimpleNamespace(spatial_merge_size=2),
        )
        self.model.language_model.config.layer_types = ["full_attention", "sliding_attention"]

    def forward(self, input_ids=None, **kwargs):
        return {"input_ids": input_ids, **kwargs}


class QwenPatchTest(unittest.TestCase):
    def test_installs_only_full_attention_and_restores_original_forward(self):
        model = _Qwen()
        session = FoveaSession(FoveaConfig())
        full_attention = model.model.language_model.layers[0].self_attn
        sliding_attention = model.model.language_model.layers[1].self_attn
        original_full = full_attention.forward
        original_sliding = sliding_attention.forward

        handle = install_tokenfovea(model, session)

        self.assertEqual(session.routed_layers, (0,))
        self.assertNotEqual(full_attention.forward, original_full)
        self.assertEqual(sliding_attention.forward, original_sliding)
        handle.remove()
        self.assertEqual(full_attention.forward, original_full)

    def test_positional_input_ids_reset_and_configure_prompt(self):
        model = _Qwen()
        session = FoveaSession(FoveaConfig())
        handle = install_tokenfovea(model, session)
        session.configure_prompt(
            torch.tensor([[1, 99, 2]]),
            torch.tensor([[1, 2, 2]]),
            image_token_id=99,
            spatial_merge_size=2,
        )

        input_ids = torch.tensor([[1, 99, 99, 99, 99, 2]])
        model(input_ids, image_grid_thw=torch.tensor([[1, 4, 4]]))

        self.assertEqual(session.prompt_length, input_ids.shape[-1])
        self.assertEqual(session.visual_positions, [1, 2, 3, 4])
        handle.remove()

    def test_rejects_non_sdpa_attention_backend(self):
        model = _Qwen()
        model.config._attn_implementation = "eager"

        with self.assertRaisesRegex(ValueError, "require attn_implementation='sdpa'"):
            install_tokenfovea(model, FoveaSession(FoveaConfig()))
