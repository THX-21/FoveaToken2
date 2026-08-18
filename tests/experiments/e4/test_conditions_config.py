from pathlib import Path

import pytest

from experiments.e4.conditions import conditions_for_suite, get_condition
from experiments.e4.config import E4Config


def test_condition_registry_has_expected_matrix():
    assert [item.name for item in conditions_for_suite("formal")] == [
        "full",
        "lowres4",
        "uniform4_native",
        "prefill_static_top8_native",
        "dynamic_top8_native",
        "dynamic_all_heads_native",
    ]
    assert len(conditions_for_suite("reasoning")) == 6
    assert len(conditions_for_suite("compression")) == 4
    assert get_condition("dynamic2_top8_native").budget_area == 4
    with pytest.raises(ValueError):
        get_condition("lowres2", "formal")


def test_default_config_locks_4096_token_caps():
    config = E4Config.load(Path("experiments/e4/configs/default.yaml"))
    assert config.models["qwen25"].max_pixels == 4096 * 28**2
    assert config.models["qwen35"].max_pixels == 4096 * 32**2
