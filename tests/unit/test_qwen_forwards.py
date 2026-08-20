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
    native_capture_scale = 4

    def __init__(self):
        self.captured = False

    def is_prefill_layer(self, layer_idx):
        return True

    def needs_signal(self, layer_idx):
        return False

    def needs_prefill_signal(self, layer_idx):
        return self.needs_signal(layer_idx)

    def capture_prefill_layer(self, *args):
        self.captured = True


class _CompactPrefillSession(_PrefillSession):
    native_capture_scale = None

    def __init__(self):
        super().__init__()
        self.recorded = False

    def compose_prefill(self, layer_idx, full_keys, full_values, reference):
        query_index = torch.tensor([full_keys.shape[-2] - 1])
        active_ids = torch.tensor([0])
        mask = torch.ones(
            1, 1, 1, full_keys.shape[-2], dtype=torch.bool
        )
        return full_keys, full_values, query_index, mask, active_ids

    def record_prefill_layer(self, layer_idx, visual_attention, active_ids):
        self.recorded = True


class QwenForwardTest(unittest.TestCase):
    def test_qwen25_native_bank_prefill_delegates_attention_to_original(self):
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
        sentinel = (torch.randn_like(hidden), torch.randn(1))
        calls = []

        def original(original_hidden, **kwargs):
            calls.append(kwargs)
            cache.update(torch.zeros(1, 1, 3, 6), torch.zeros(1, 1, 3, 6), 0)
            return sentinel

        position_embeddings = (
            torch.ones(3, 1, 3, hidden_size),
            torch.zeros(3, 1, 3, hidden_size),
        )

        result = make_qwen25_forward(original, session)(
            module,
            hidden,
            past_key_values=cache,
            use_cache=True,
            position_embeddings=position_embeddings,
        )

        self.assertIs(result, sentinel)
        self.assertEqual(len(calls), 1)
        self.assertTrue(session.captured)
        self.assertEqual(len(cache.updates), 1)

    def test_qwen25_main_prefill_replaces_post_image_text_attention(self):
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
            config=types.SimpleNamespace(
                rope_parameters={"mrope_section": [1, 1, 1]}
            ),
        )
        session = _CompactPrefillSession()
        cache = _Cache()
        hidden = torch.randn(1, 3, hidden_size)
        original_output = torch.full_like(hidden, 99.0)

        def original(original_hidden, **kwargs):
            cache.update(
                original_hidden.view(1, 1, 3, 6),
                original_hidden.view(1, 1, 3, 6),
                0,
            )
            return original_output, None

        position_embeddings = (
            torch.ones(3, 1, 3, hidden_size),
            torch.zeros(3, 1, 3, hidden_size),
        )
        output, _ = make_qwen25_forward(original, session)(
            module,
            hidden,
            past_key_values=cache,
            use_cache=True,
            position_embeddings=position_embeddings,
        )

        self.assertTrue(torch.equal(output[:, :2], original_output[:, :2]))
        self.assertFalse(torch.equal(output[:, 2:], original_output[:, 2:]))
        self.assertTrue(session.recorded)
        self.assertEqual(len(cache.updates), 1)

    def test_qwen35_native_bank_prefill_delegates_attention_to_original(self):
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
        sentinel = (torch.randn_like(hidden), torch.randn(1))
        calls = []

        def original(original_hidden, **kwargs):
            calls.append(kwargs)
            cache.update(torch.zeros(1, 1, 3, 4), torch.zeros(1, 1, 3, 4), 0)
            return sentinel

        position_embeddings = (
            torch.ones(1, 3, hidden_size),
            torch.zeros(1, 3, hidden_size),
        )

        result = make_qwen35_forward(original, session)(
            module,
            hidden,
            position_embeddings=position_embeddings,
            attention_mask=None,
            past_key_values=cache,
            use_cache=True,
        )

        self.assertIs(result, sentinel)
        self.assertEqual(len(calls), 1)
        self.assertTrue(session.captured)
        self.assertEqual(len(cache.updates), 1)

    def test_qwen35_main_prefill_replaces_post_image_text_attention(self):
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
        session = _CompactPrefillSession()
        cache = _Cache()
        hidden = torch.randn(1, 3, hidden_size)
        original_output = torch.full_like(hidden, 99.0)

        def original(original_hidden, **kwargs):
            cache.update(
                original_hidden.view(1, 1, 3, 4),
                original_hidden.view(1, 1, 3, 4),
                0,
            )
            return original_output, None

        position_embeddings = (
            torch.ones(1, 3, hidden_size),
            torch.zeros(1, 3, hidden_size),
        )
        output, _ = make_qwen35_forward(original, session)(
            module,
            hidden,
            position_embeddings=position_embeddings,
            attention_mask=None,
            past_key_values=cache,
            use_cache=True,
        )

        self.assertTrue(torch.equal(output[:, :2], original_output[:, :2]))
        self.assertFalse(torch.equal(output[:, 2:], original_output[:, 2:]))
        self.assertTrue(session.recorded)
        self.assertEqual(len(cache.updates), 1)


if __name__ == "__main__":
    unittest.main()
