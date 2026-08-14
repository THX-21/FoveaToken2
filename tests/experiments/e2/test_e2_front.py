import unittest

import torch

from experiments.e2.conditions import CONDITIONS
from experiments.e2.front import BlockFront, stable_seed


class BlockFrontTest(unittest.TestCase):
    def test_e2_registers_twenty_conditions_with_four_native_variants(self):
        names = {condition.name for condition in CONDITIONS}
        self.assertEqual(len(CONDITIONS), 20)
        self.assertTrue(
            {
                "native_uniform4",
                "native_uniform16",
                "random_fixed_native",
                "random_perstep_native",
            }.issubset(names)
        )

    def test_uniform_fronts_cover_grid_and_pool_means(self):
        values = torch.arange(64, dtype=torch.float32)
        two = BlockFront.uniform(8, 8, 2)
        four = BlockFront.uniform(8, 8, 4)
        eight = BlockFront.uniform(8, 8, 8)

        self.assertEqual(two.node_count, 16)
        self.assertEqual(four.node_count, 4)
        self.assertEqual(eight.node_count, 1)
        self.assertAlmostEqual(float(two.pool(values, 0)[0]), 4.5)
        self.assertAlmostEqual(float(four.pool(values, 0)[0]), 13.5)
        self.assertAlmostEqual(float(eight.pool(values, 0)[0]), 31.5)

    def test_random_front_is_reproducible_and_preserves_scale_counts(self):
        first = BlockFront.random_multiscale(20, 20, stable_seed(42, "sample"))
        second = BlockFront.random_multiscale(20, 20, stable_seed(42, "sample"))
        other = BlockFront.random_multiscale(20, 20, stable_seed(42, "sample", 1))

        self.assertEqual(first.digest(), second.digest())
        self.assertNotEqual(first.digest(), other.digest())
        self.assertEqual(first.scale_counts, other.scale_counts)
        macroblocks = 25
        self.assertEqual(first.scale_counts[1], 13 * 16)
        self.assertEqual(first.scale_counts[2], 7 * 4)
        self.assertEqual(first.scale_counts[4], macroblocks - 13 - 7)

    def test_pool_supports_arbitrary_node_dimension(self):
        front = BlockFront.uniform(4, 4, 2)
        values = torch.arange(32, dtype=torch.float32).reshape(1, 16, 2)

        pooled = front.pool(values, 1)

        self.assertEqual(pooled.shape, (1, 4, 2))
        self.assertTrue(torch.allclose(pooled[0, 0], values[0, [0, 1, 4, 5]].mean(0)))


if __name__ == "__main__":
    unittest.main()
