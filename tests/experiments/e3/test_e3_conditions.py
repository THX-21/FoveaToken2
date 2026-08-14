import unittest

from experiments.e3.conditions import CONDITIONS, PAIRS, get_condition
from experiments.e3.config import E3Config


class E3ConditionsTest(unittest.TestCase):
    def test_registers_four_control_anchor_pairs(self):
        self.assertEqual(len(CONDITIONS), 8)
        self.assertEqual(
            PAIRS,
            {
                "full_text_anchor": "full_mrope",
                "lowres2_text_anchor": "lowres2_mrope",
                "pool2_text_anchor": "pool2_center",
                "native2_text_anchor": "native2_center",
            },
        )
        self.assertFalse(get_condition("full_mrope").text_anchor)
        self.assertFalse(get_condition("lowres2_mrope").compact)
        self.assertTrue(get_condition("pool2_center").compact)
        self.assertTrue(get_condition("native2_center").native)

    def test_default_config_reuses_e2_manifest(self):
        config = E3Config.load("experiments/e3/configs/default.yaml")
        self.assertEqual(config.sample_count, 100)
        self.assertEqual(config.max_new_tokens, 320)
        self.assertEqual(config.anchor_window, 8.0)
        self.assertEqual(str(config.e2_data_dir), "data/e2")


if __name__ == "__main__":
    unittest.main()
