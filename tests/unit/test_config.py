import unittest

from tokenfovea.config import FoveaConfig


class ConfigTest(unittest.TestCase):
    def test_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            FoveaConfig(budget=0)
        with self.assertRaises(ValueError):
            FoveaConfig(mode="oracle")
        with self.assertRaises(ValueError):
            FoveaConfig(attention_ema=1.0)
        with self.assertRaises(ValueError):
            FoveaConfig(position_mode="bad")
        with self.assertRaises(ValueError):
            FoveaConfig(pooling_mode="bad")
        with self.assertRaises(ValueError):
            FoveaConfig(pooling_mode="hidden", position_mode="post_rope_pool")
