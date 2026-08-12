#!/usr/bin/env python3
from __future__ import annotations

import torch

from experiments.e1.metrics import gaze_statistics, hybrid_statistics, visual_statistics
from experiments.e1.probe import full_context_attention


def main() -> None:
    torch.manual_seed(42)
    query = torch.randn(1, 4, 1, 8)
    key = torch.randn(1, 2, 12, 8)
    attention = full_context_attention(query, key, groups=2, scaling=8**-0.5)
    assert attention.shape == (1, 4, 12)
    assert torch.allclose(attention.sum(dim=-1), torch.ones(1, 4), atol=1e-6)
    mass, concentration, _ = visual_statistics(attention[0, :, 2:10])
    assert torch.all((0 <= mass) & (mass <= 1))
    assert torch.all((0 <= concentration) & (concentration <= 1))

    static = torch.zeros(1, 20, dtype=torch.int32)
    static[0, :2] = 10
    dynamic = torch.ones(1, 20, dtype=torch.int32)
    static_stats = hybrid_statistics(static, 10)[0]
    dynamic_stats = hybrid_statistics(dynamic, 10)[0]
    assert static_stats.coverage < dynamic_stats.coverage
    assert static_stats.persistence > dynamic_stats.persistence

    matrix = torch.eye(9).unsqueeze(0)
    gaze = gaze_statistics(matrix, torch.zeros(1, 9))[0]
    assert gaze.calibrated_score == 1.0
    print("E1 CPU smoke test passed")


if __name__ == "__main__":
    main()
