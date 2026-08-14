from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Pooling = Literal["kv", "hidden", "native_multiscale"]
Position = Literal["native_center", "post_rope_pool"]
FrontMode = Literal["full", "uniform", "random_fixed", "random_perstep", "lowres"]


@dataclass(frozen=True, slots=True)
class Condition:
    name: str
    front_mode: FrontMode
    pooling: Pooling = "kv"
    position: Position = "native_center"
    block_size: int | None = None
    lowres_divisor: int | None = None

    @property
    def pooled(self) -> bool:
        return self.front_mode in {"uniform", "random_fixed", "random_perstep"}

    @property
    def native(self) -> bool:
        return self.pooling == "native_multiscale"


CONDITIONS = (
    Condition("full", "full"),
    Condition("lowres_2", "lowres", lowres_divisor=2),
    Condition("uniform2_kv_center", "uniform", block_size=2),
    Condition("uniform2_hidden_center", "uniform", pooling="hidden", block_size=2),
    Condition("uniform2_postrope", "uniform", position="post_rope_pool", block_size=2),
    Condition("native_uniform4", "uniform", pooling="native_multiscale", block_size=2),
    Condition("lowres_4", "lowres", lowres_divisor=4),
    Condition("uniform4_kv_center", "uniform", block_size=4),
    Condition("uniform4_hidden_center", "uniform", pooling="hidden", block_size=4),
    Condition("uniform4_postrope", "uniform", position="post_rope_pool", block_size=4),
    Condition("native_uniform16", "uniform", pooling="native_multiscale", block_size=4),
    Condition("random_fixed_kv_center", "random_fixed"),
    Condition("random_fixed_hidden_center", "random_fixed", pooling="hidden"),
    Condition("random_fixed_postrope", "random_fixed", position="post_rope_pool"),
    Condition("random_fixed_native", "random_fixed", pooling="native_multiscale"),
    Condition("random_perstep_kv_center", "random_perstep"),
    Condition("random_perstep_hidden_center", "random_perstep", pooling="hidden"),
    Condition("random_perstep_postrope", "random_perstep", position="post_rope_pool"),
    Condition("random_perstep_native", "random_perstep", pooling="native_multiscale"),
    Condition("lowres_random_matched", "lowres"),
)

CONDITION_BY_NAME = {condition.name: condition for condition in CONDITIONS}


def get_condition(name: str) -> Condition:
    try:
        return CONDITION_BY_NAME[name]
    except KeyError as error:
        raise ValueError(f"unknown E2 condition {name!r}; choose from {sorted(CONDITION_BY_NAME)}") from error
