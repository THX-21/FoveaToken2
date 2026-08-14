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
        with self.assertRaises(ValueError):
            FoveaConfig(pooling_mode="native_multiscale", position_mode="text_anchor")

    def test_accepts_native_multiscale_with_native_center(self):
        config = FoveaConfig(pooling_mode="native_multiscale")
        self.assertEqual(config.pooling_mode, "native_multiscale")
