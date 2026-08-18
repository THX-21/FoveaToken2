import json

import pytest
from PIL import Image

from experiments.e4.image import aligned_high_resolution, scaled_plan, visual_tokens
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
