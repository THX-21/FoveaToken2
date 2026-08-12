import unittest

import torch

from tokenfovea.router import SplitMergeRouter
from tokenfovea.topology import DeviceTreeTopology, VisualTokenForest


class RouterTest(unittest.TestCase):
    def test_preserves_budget_and_refines_high_score_region(self):
        forest = VisualTokenForest.from_grids([(8, 8)])
        topology = DeviceTreeTopology.build(forest, torch.device("cpu"))
        initial = torch.tensor(sorted(forest.initial_front(16)))
        router = SplitMergeRouter(topology, initial, epsilon=0.0, max_swaps=1)

        def covering_size(active_ids: torch.Tensor, leaf: int) -> int:
            return next(forest.node(i).valid_count for i in active_ids.tolist() if leaf in forest.node(i).leaf_indices)

        before_ids = router.active_ids()
        leaf_scores = torch.zeros(forest.num_leaves)
        leaf_scores[0] = 1.0
        swaps = router.step(topology.aggregate_leaves(leaf_scores, density=False))
        after_ids = router.active_ids()

        self.assertEqual(int(swaps), 1)
        self.assertEqual(after_ids.numel(), before_ids.numel())
        self.assertLess(covering_size(after_ids, 0), covering_size(before_ids, 0))
        forest.validate_front(after_ids.tolist())

    def test_preserves_random_ragged_fronts(self):
        generator = torch.Generator().manual_seed(7)
        for grid in ((7, 5), (4, 6), (8, 8)):
            forest = VisualTokenForest.from_grids([grid])
            topology = DeviceTreeTopology.build(forest, torch.device("cpu"))
            for budget in (4, 11, min(24, forest.num_leaves)):
                initial = torch.tensor(sorted(forest.initial_front(budget)))
                for score_mode in ("mass", "density"):
                    router = SplitMergeRouter(
                        topology,
                        initial,
                        epsilon=0.0,
                        max_swaps=8,
                        score_mode=score_mode,
                    )
                    leaf_scores = torch.rand(forest.num_leaves, generator=generator)
                    router.step(topology.aggregate_leaves(leaf_scores, density=score_mode == "density"))
                    active = router.active_ids()
                    self.assertEqual(active.numel(), initial.numel())
                    forest.validate_front(active.tolist())
