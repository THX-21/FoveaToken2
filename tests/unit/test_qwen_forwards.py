import types
import unittest

import torch

from tokenfovea.integrations.qwen.qwen25_vl import make_forward as make_qwen25_forward
from tokenfovea.integrations.qwen.qwen35 import make_forward as make_qwen35_forward


class _Cache:
    def __init__(self):
        self.updates = []
        self.lengths = {}

    def update(self, key, value, layer_idx):
        self.updates.append((key, value, layer_idx))
        self.lengths[layer_idx] = key.shape[-2]
        return key, value

    def get_seq_length(self, layer_idx=0):
        return self.lengths.get(layer_idx, 0)


class _PrefillSession:
    is_configured = True
    enabled = True
    config = types.SimpleNamespace(pooling_mode="kv")

    def __init__(self):
        self.captured = False

    def is_prefill_layer(self, layer_idx):
        return True

    def needs_signal(self, layer_idx):
        return False

    def capture_prefill_layer(self, *args):
        self.captured = True


def _unexpected_original(*args, **kwargs):
    raise AssertionError("prefill should reuse projected Q/K/V")


class QwenForwardTest(unittest.TestCase):
    def test_qwen25_prefill_reuses_projected_qkv(self):
        hidden_size = 6
        module = types.SimpleNamespace(
            q_proj=torch.nn.Identity(),
            k_proj=torch.nn.Identity(),
            v_proj=torch.nn.Identity(),
            o_proj=torch.nn.Identity(),
            head_dim=hidden_size,
            num_key_value_groups=1,
            scaling=hidden_size**-0.5,
            layer_idx=0,
            config=types.SimpleNamespace(rope_parameters={"mrope_section": [1, 1, 1]}),
        )
        session = _PrefillSession()
        cache = _Cache()
        hidden = torch.randn(1, 3, hidden_size)
        position_embeddings = (
            torch.ones(3, 1, 3, hidden_size),
            torch.zeros(3, 1, 3, hidden_size),
        )

        output, weights = make_qwen25_forward(_unexpected_original, session)(
            module,
            hidden,
            past_key_values=cache,
            use_cache=True,
            position_embeddings=position_embeddings,
        )

        self.assertEqual(output.shape, hidden.shape)
        self.assertIsNone(weights)
        self.assertTrue(session.captured)
        self.assertEqual(len(cache.updates), 1)

    def test_qwen35_prefill_reuses_projected_qkv(self):
        hidden_size = 4
        module = types.SimpleNamespace(
            q_proj=torch.nn.Linear(hidden_size, hidden_size * 2, bias=False),
            k_proj=torch.nn.Identity(),
            v_proj=torch.nn.Identity(),
            o_proj=torch.nn.Identity(),
            q_norm=torch.nn.Identity(),
            k_norm=torch.nn.Identity(),
            head_dim=hidden_size,
            num_key_value_groups=1,
            scaling=hidden_size**-0.5,
            layer_idx=0,
        )
        session = _PrefillSession()
        cache = _Cache()
        hidden = torch.randn(1, 3, hidden_size)
        position_embeddings = (
            torch.ones(1, 3, hidden_size),
            torch.zeros(1, 3, hidden_size),
        )

        output, weights = make_qwen35_forward(_unexpected_original, session)(
            module,
            hidden,
            position_embeddings=position_embeddings,
            attention_mask=None,
            past_key_values=cache,
            use_cache=True,
        )

        self.assertEqual(output.shape, hidden.shape)
        self.assertIsNone(weights)
        self.assertTrue(session.captured)
        self.assertEqual(len(cache.updates), 1)


if __name__ == "__main__":
    unittest.main()
