import unittest

import torch

from tokenfovea.integrations.qwen.common import attention, validate_cache_result, visual_attention_signal


class QwenCommonTest(unittest.TestCase):
    def test_rejects_preallocated_cache_results(self):
        cache = type("Cache", (), {"get_seq_length": lambda self, layer_idx: 3})()
        keys = torch.zeros(1, 1, 8, 4)

        with self.assertRaisesRegex(ValueError, "only populated K/V entries"):
            validate_cache_result(cache, keys, 0)

    def test_sdpa_gqa_matches_eager_attention(self):
        torch.manual_seed(3)
        query = torch.randn(1, 4, 1, 8)
        key = torch.randn(1, 2, 7, 8)
        value = torch.randn(1, 2, 7, 8)
        fast, _ = attention(query, key, value, 2, 8**-0.5, False)
        eager, _ = attention(query, key, value, 2, 8**-0.5, True)
        signal = visual_attention_signal(query, key[..., :3, :], 2, 8**-0.5)
        self.assertTrue(torch.allclose(fast, eager, atol=2e-6))
        self.assertEqual(signal.shape, (4, 3))
        self.assertTrue(torch.allclose(signal.sum(dim=-1), torch.ones(4)))

    def test_multitoken_attention_uses_bottom_right_causal_mask(self):
        query = torch.zeros(1, 1, 3, 1)
        key = torch.zeros(1, 1, 5, 1)
        value = torch.arange(5, dtype=torch.float32).view(1, 1, 5, 1)

        output, _ = attention(query, key, value, 1, 1.0, False)

        expected = torch.tensor([[[[1.0]], [[1.5]], [[2.0]]]])
        self.assertTrue(torch.allclose(output, expected))

    def test_prefill_sdpa_matches_eager_causal_attention(self):
        query = torch.zeros(1, 1, 3, 1)
        key = torch.zeros(1, 1, 3, 1)
        value = torch.arange(3, dtype=torch.float32).view(1, 1, 3, 1)

        fast, _ = attention(query, key, value, 1, 1.0, False)
        eager, _ = attention(query, key, value, 1, 1.0, True)

        expected = torch.tensor([[[[0.0]], [[0.5]], [[1.0]]]])
        self.assertTrue(torch.allclose(fast, expected))
        self.assertTrue(torch.allclose(eager, expected))
