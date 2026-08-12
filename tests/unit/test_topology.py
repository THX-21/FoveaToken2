import unittest

from tokenfovea.topology import VisualTokenForest


class TopologyTest(unittest.TestCase):
    def test_rectangular_forest_is_an_exact_cover(self):
        forest = VisualTokenForest.from_grids([(7, 5), (4, 6)])
        self.assertEqual(forest.num_leaves, 59)
        for budget in (2, 7, 16, 31, 59):
            active = forest.initial_front(budget)
            forest.validate_front(active)
