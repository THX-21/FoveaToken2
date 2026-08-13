from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from experiments.e2.conditions import get_condition
from experiments.e2.front import BlockFront
from experiments.e2.session import E2Session


def main() -> None:
    uniform = BlockFront.uniform(8, 12, 2)
    random = BlockFront.random_multiscale(8, 12, seed=42)
    values = torch.arange(96, dtype=torch.float32)
    assert uniform.pool(values, 0).numel() == 24
    assert random.pool(values, 0).numel() == random.node_count
    with tempfile.TemporaryDirectory() as directory:
        session = E2Session(
            get_condition("random_fixed_kv_center"), trace_path=Path(directory) / "trace.jsonl"
        )
        session.begin_sample("smoke")
    print("E2 CPU smoke test passed")


if __name__ == "__main__":
    main()
