import unittest

import torch

from tokenfovea.integrations.qwen.common import attention, visual_attention_signal


class QwenCommonTest(unittest.TestCase):
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
