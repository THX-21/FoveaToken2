import unittest

from tokenfovea.topology import VisualTokenForest


class TopologyTest(unittest.TestCase):
    def test_rectangular_forest_is_an_exact_cover(self):
        forest = VisualTokenForest.from_grids([(7, 5), (4, 6)])
        self.assertEqual(forest.num_leaves, 59)
        for budget in (2, 7, 16, 31, 59):
            active = forest.initial_front(budget)
            forest.validate_front(active)

    def test_aligned_forest_contains_only_native_square_scales(self):
        forest = VisualTokenForest.from_aligned_grids([(8, 16), (8, 8)])

        self.assertEqual(forest.num_leaves, 192)
        self.assertEqual(len(forest.roots), 3)
        self.assertEqual(
            {(node.y1 - node.y0, node.x1 - node.x0) for node in forest.nodes},
            {(1, 1), (2, 2), (4, 4), (8, 8)},
        )
        forest.validate_front(forest.roots)

    def test_aligned_forest_rejects_non_aligned_grids(self):
        with self.assertRaisesRegex(ValueError, "divisible by 8"):
            VisualTokenForest.from_aligned_grids([(8, 12)])

    def test_aligned_arbitrary_budget_is_exact_and_spatially_distributed(self):
        forest = VisualTokenForest.from_aligned_grids([(16, 16)])
        active = forest.initial_front(31)
        self.assertEqual(len(active), 31)
        forest.validate_front(active)
        refined = [forest.node(node_id) for node_id in active if forest.node(node_id).valid_count == 4]
        self.assertEqual(len(refined), 20)
        self.assertEqual(
            {node.y0 < 8 for node in refined},
            {False, True},
        )
        self.assertEqual(
            {node.x0 < 8 for node in refined},
            {False, True},
        )
