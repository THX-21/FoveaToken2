import json

import pytest
from PIL import Image

from experiments.e4.image import (
    aligned_high_resolution,
    matched_budget_plan,
    scaled_plan,
    visual_tokens,
)
from experiments.e4.runtime import validate_head_selection


@pytest.mark.parametrize("size", [(8000, 1000), (1000, 8000), (4000, 3000)])
def test_highres_plan_is_capped_and_aligned(size):
    plan = aligned_high_resolution(Image.new("RGB", size), 28, 200704, 4096)
    assert plan.grid_width % 8 == plan.grid_height % 8 == 0
    assert visual_tokens(plan) <= 4096
    assert visual_tokens(scaled_plan(plan, 2)) == visual_tokens(plan) // 4
    assert visual_tokens(scaled_plan(plan, 4)) == visual_tokens(plan) // 16
    assert visual_tokens(scaled_plan(plan, 8)) == visual_tokens(plan) // 64


def test_top8_selection_validation(tmp_path):
    path = tmp_path / "heads.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "model": "model",
                "selected_heads": [
                    {"layer": index // 2, "head": index % 2} for index in range(8)
                ],
            }
        ),
        encoding="utf-8",
    )
    payload = validate_head_selection(
        path,
        "model",
        routed_layers={0, 1, 2, 3},
        heads_per_layer={0: 2, 1: 2, 2: 2, 3: 2},
    )
    assert len(payload["selected_heads"]) == 8
    with pytest.raises(ValueError):
        validate_head_selection(path, "other")


@pytest.mark.parametrize("ratio", [6.0, 7.5, 8.0, 64.0])
def test_arbitrary_ratio_plan_matches_lowres_and_native_budget(ratio):
    plans = [
        aligned_high_resolution(Image.new("RGB", (913, 507)), 28, 200704, 4096),
        aligned_high_resolution(Image.new("RGB", (507, 913)), 28, 200704, 4096),
    ]
    matched = matched_budget_plan(plans, ratio)
    high_tokens = sum(visual_tokens(plan) for plan in plans)
    roots = sum(visual_tokens(plan) // 64 for plan in plans)
    lowres_tokens = sum(visual_tokens(plan) for plan in matched.lowres_plans)
    assert lowres_tokens == matched.target_tokens
    assert (matched.target_tokens - roots) % 3 == 0
    assert matched.achieved_ratio == high_tokens / lowres_tokens
    assert all(plan.grid_height % 8 == plan.grid_width % 8 == 0 for plan in plans)


def test_ratio_eight_plan_is_not_a_fixed_spatial_divisor():
    high = aligned_high_resolution(Image.new("RGB", (1024, 1024)), 28, 200704, 4096)
    low = matched_budget_plan([high], 8.0).lowres_plans[0]
    assert (low.grid_height, low.grid_width) == (14, 14)
    assert visual_tokens(low) == 196
    assert high.grid_height / low.grid_height != 8
