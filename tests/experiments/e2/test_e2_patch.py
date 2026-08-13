import unittest
import types

import torch

from experiments.e2.conditions import get_condition
from experiments.e2.patch import _compact_attention, install_e2
from experiments.e2.session import E2Session


class E2CompactAttentionTest(unittest.TestCase):
    def test_uses_explicit_original_position_mask(self):
        query = torch.zeros(1, 1, 2, 1)
        key = torch.zeros(1, 1, 4, 1)
        value = torch.arange(4, dtype=torch.float32).view(1, 1, 4, 1)
        mask = torch.tensor([[[[True, True, False, False], [True, True, True, True]]]])

        output = _compact_attention(query, key, value, 1, 1.0, mask)

        expected = torch.tensor([[[[0.5]], [[1.5]]]])
        self.assertTrue(torch.allclose(output, expected))

    def test_expands_gqa_heads(self):
        torch.manual_seed(5)
        query = torch.randn(1, 4, 2, 3)
        key = torch.randn(1, 2, 5, 3)
        value = torch.randn(1, 2, 5, 3)
        mask = torch.ones(1, 1, 2, 5, dtype=torch.bool)

        actual = _compact_attention(query, key, value, 2, 3**-0.5, mask)
        repeated_key = key.repeat_interleave(2, 1)
        repeated_value = value.repeat_interleave(2, 1)
        weights = torch.softmax(torch.matmul(query, repeated_key.transpose(-2, -1)) * 3**-0.5, -1)
        expected = torch.matmul(weights, repeated_value).transpose(1, 2)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_installer_patches_only_full_attention_layers(self):
        class Attention(torch.nn.Module):
            def forward(self, hidden_states, **kwargs):
                return hidden_states, None

        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = Attention()

        layers = torch.nn.ModuleList([Layer(), Layer()])
        language = torch.nn.Module()
        language.layers = layers
        language.config = types.SimpleNamespace(layer_types=["full_attention", "linear_attention"], rope_scaling={"mrope_section": [1]})
        language.rotary_emb = lambda reference, positions: (torch.ones_like(reference), torch.zeros_like(reference))
        model = torch.nn.Module()
        model.model = types.SimpleNamespace(language_model=language)
        model.config = types.SimpleNamespace(
            model_type="qwen3_5", _attn_implementation="sdpa", image_token_id=99,
            vision_config=types.SimpleNamespace(spatial_merge_size=2),
        )
        original_linear = layers[1].self_attn.forward
        session = E2Session(get_condition("random_fixed_kv_center"))

        handle = install_e2(model, session)
        try:
            self.assertEqual(session.routed_layers, (0,))
            self.assertEqual(layers[1].self_attn.forward, original_linear)
        finally:
            handle.remove()


if __name__ == "__main__":
    unittest.main()
