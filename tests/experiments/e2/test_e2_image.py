import unittest

from PIL import Image

from experiments.e2.image import aligned_high_resolution, lowres_plan, matched_lowres_plan


class E2ImagePlanTest(unittest.TestCase):
    def test_aligned_grid_and_uniform_lowres_match_exactly(self):
        image = Image.new("RGB", (913, 507))
        high = aligned_high_resolution(image, 28, 200704, 1605632)

        self.assertEqual(high.grid_height % 4, 0)
        self.assertEqual(high.grid_width % 4, 0)
        self.assertEqual(lowres_plan(high, 2).visual_tokens, high.visual_tokens // 4)
        self.assertEqual(lowres_plan(high, 4).visual_tokens, high.visual_tokens // 16)

    def test_random_budget_matching_prefers_exact_token_count(self):
        image = Image.new("RGB", (800, 600))
        high = aligned_high_resolution(image, 32, 65536, 131072)
        plan = matched_lowres_plan(high, 37, 32)

        self.assertLessEqual(abs(plan.visual_tokens - 37), 1)
        self.assertLessEqual(plan.grid_width, high.grid_width)
        self.assertLessEqual(plan.grid_height, high.grid_height)

    def test_scale64_lowres_plan_is_supported_for_aligned_inputs(self):
        image = Image.new("RGB", (1024, 1024))
        high = aligned_high_resolution(image, 32, 65536, 1048576)
        if high.grid_width % 8 or high.grid_height % 8:
            self.skipTest("chosen E2 plan is intentionally only four-aligned")
        self.assertEqual(lowres_plan(high, 8).visual_tokens, high.visual_tokens // 64)


if __name__ == "__main__":
    unittest.main()
