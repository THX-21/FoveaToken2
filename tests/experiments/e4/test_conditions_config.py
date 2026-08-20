from pathlib import Path

import pytest

from experiments.e4.conditions import conditions_for_suite, get_condition
from experiments.e4.config import E4Config


def test_condition_registry_has_expected_matrix():
    assert [item.name for item in conditions_for_suite("formal")] == [
        "full",
        "lowres8",
        "uniform8_native",
        "prefill_static8_top8_native",
        "dynamic8_top8_native",
        "dynamic8_all_heads_native",
    ]
    assert len(conditions_for_suite("reasoning")) == 6
    compression = conditions_for_suite("compression")
    assert len(compression) == 20
    assert {item.compression_ratio for item in compression} == {2, 4, 6, 8, 16}
    assert get_condition(
        "dynamic6p5_top8_native",
        compression_ratio=6.5,
        compression_ratios=(6.5,),
    ).routed
    with pytest.raises(ValueError):
        get_condition("lowres4", "formal")


def test_default_config_locks_4096_token_caps():
    config = E4Config.load(Path("experiments/e4/configs/default.yaml"))
    assert config.compression_ratio == 8
    assert config.compression_ratios == (2, 4, 6, 8, 16)
    assert config.models["qwen25"].max_pixels == 4096 * 28**2
    assert config.models["qwen35"].max_pixels == 4096 * 32**2
