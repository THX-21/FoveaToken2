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
            FoveaConfig(pooling_mode="native_multiscale", position_mode="post_rope_pool")

    def test_accepts_native_multiscale_position_modes(self):
        for position_mode in ("native_center", "text_anchor", "no_rope"):
            with self.subTest(position_mode=position_mode):
                config = FoveaConfig(
                    pooling_mode="native_multiscale",
                    position_mode=position_mode,
                )
                self.assertEqual(config.position_mode, position_mode)
