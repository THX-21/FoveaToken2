import types
import unittest

import torch

from experiments.e1.probe import E1AttentionProbe, full_context_attention


class _Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = 6
        self.num_key_value_groups = 2
        self.scaling = 6**-0.5
        self.q_proj = torch.nn.Linear(6, 12, bias=False)
        self.k_proj = torch.nn.Linear(6, 6, bias=False)
        self.config = types.SimpleNamespace(rope_scaling={"mrope_section": [1, 1, 1]})

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None):
        return hidden_states * 2


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()


class _Language(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer(), _Layer()])
        self.config = types.SimpleNamespace(
            layer_types=["full_attention", "linear_attention"],
            rope_scaling={"mrope_section": [1, 1, 1]},
        )


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = types.SimpleNamespace(language_model=_Language())
        self.config = types.SimpleNamespace(
            model_type="qwen2_5_vl",
            image_token_id=99,
            vision_config=types.SimpleNamespace(spatial_merge_size=2),
        )


class E1ProbeTest(unittest.TestCase):
    def test_full_context_attention_matches_direct_eager_gqa_with_mask(self):
        torch.manual_seed(4)
        query = torch.randn(1, 4, 2, 8)
        key = torch.randn(1, 2, 5, 8)
        mask = torch.tensor([[1, 1, 1, 1, 0]])

        actual = full_context_attention(query, key, 2, 8**-0.5, mask)
        repeated = key.repeat_interleave(2, dim=1)
        logits = torch.matmul(query[..., -1:, :], repeated.transpose(-2, -1)) * 8**-0.5
        logits[..., -1] = -torch.inf
        expected = torch.softmax(logits.float(), dim=-1).squeeze(-2)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        self.assertTrue(torch.all(actual[..., -1] == 0))

    def test_probe_excludes_linear_layers_and_does_not_change_forward(self):
        model = _Model()
        attention = model.model.language_model.layers[0].self_attn
        hidden = torch.randn(1, 3, 6)
        cos = torch.ones(3, 1, 3, 6)
        sin = torch.zeros_like(cos)
        expected = attention(hidden, position_embeddings=(cos, sin))
        probe = E1AttentionProbe(model, "/tmp/tokenfovea-e1-test")
        try:
            self.assertEqual(probe.routed_layers, (0,))
            probe.begin_sample(
                "sample",
                "natural",
                torch.tensor([[1, 99, 2]]),
                torch.tensor([[1, 2, 2]]),
            )
            actual = attention(hidden, position_embeddings=(cos, sin))
            decode_hidden = torch.randn(1, 1, 6)
            decode_cos = torch.ones(3, 1, 1, 6)
            decode_sin = torch.zeros_like(decode_cos)
            attention(
                decode_hidden,
                attention_mask=torch.ones(1, 4),
                position_embeddings=(decode_cos, decode_sin),
            )
            self.assertEqual(probe._keys[0].shape[-2], 4)
            probe.end_sample()
            natural, _, metadata = probe.summaries()
        finally:
            probe.handle.remove()

        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(metadata["full_attention_layers"], [0])
        self.assertEqual(len(natural), 2)


if __name__ == "__main__":
    unittest.main()
