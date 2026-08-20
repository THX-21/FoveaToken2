from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
from PIL import Image

from experiments.e4.conditions import conditions_for_suite
from experiments.e4.evaluator import parse_reasoning_response
from experiments.e4.image import aligned_high_resolution, matched_budget_plan, visual_tokens
from experiments.e4.runtime import RouteTraceObserver, validate_head_selection
from tokenfovea.session import RouteEvent
from tokenfovea.topology import VisualTokenForest


def main() -> None:
    plan = aligned_high_resolution(Image.new("RGB", (2048, 1024)), 28, 200704, 4096)
    budget = matched_budget_plan([plan], 8.0)
    assert visual_tokens(budget.lowres_plans[0]) == budget.target_tokens
    assert len(conditions_for_suite("formal")) == 6
    forest = VisualTokenForest.from_aligned_grids([(8, 8)])
    observer = RouteTraceObserver()
    observer.begin_sample("smoke")
    observer(
        RouteEvent(
            "prefill",
            0,
            True,
            torch.tensor([forest.roots[0]]),
            torch.tensor([forest.roots[0]]),
            torch.ones(len(forest.nodes)),
            torch.tensor(0),
            forest,
        )
    )
    assert observer.drain()[0]["active_after"][0]["area_scale"] == 64
    assert parse_reasoning_response("Analyze. <answer>A</answer>")[0] == "A"
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "heads.json"
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "model": "smoke",
                    "selected_heads": [
                        {"layer": index, "head": 0} for index in range(8)
                    ],
                }
            ),
            encoding="utf-8",
        )
        validate_head_selection(path, "smoke")
    print("E4 CPU smoke passed")


if __name__ == "__main__":
    main()
