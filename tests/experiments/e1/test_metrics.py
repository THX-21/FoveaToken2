import unittest

import torch

from experiments.e1.metrics import gaze_statistics, hybrid_statistics, visual_statistics


class E1MetricsTest(unittest.TestCase):
    def test_visual_statistics_keep_full_context_mass(self):
        visual = torch.tensor([[0.20, 0.10], [0.15, 0.15]])
        mass, concentration, distribution = visual_statistics(visual)

        self.assertTrue(torch.allclose(mass, torch.tensor([0.30, 0.30])))
        self.assertTrue(torch.allclose(distribution.sum(dim=-1), torch.ones(2)))
        self.assertGreater(float(concentration[0]), float(concentration[1]))

    def test_hybrid_statistics_separate_static_and_dynamic(self):
        counts = torch.tensor(
            [
                [10, 10, 0, 0, 0, 0, 0, 0, 0, 0],
                [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            ],
            dtype=torch.int32,
        )

        static, dynamic = hybrid_statistics(counts, step_count=10)

        self.assertLess(static.coverage, dynamic.coverage)
        self.assertGreater(static.persistence, dynamic.persistence)

    def test_gaze_score_rewards_diagonal_not_fixed_or_uniform_attention(self):
        diagonal = torch.eye(9)
        fixed = torch.zeros(9, 9)
        fixed[:, 0] = 1
        uniform = torch.full((9, 9), 1 / 9)
        matrix = torch.stack((diagonal, fixed, uniform))
        null = torch.zeros(3, 9)

        scores = gaze_statistics(matrix, null)

        self.assertEqual(scores[0].raw_score, 1.0)
        self.assertAlmostEqual(scores[1].raw_score, 1 / 9)
        self.assertAlmostEqual(scores[2].raw_score, 1 / 9)


if __name__ == "__main__":
    unittest.main()
