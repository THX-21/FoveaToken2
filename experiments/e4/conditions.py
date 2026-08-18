from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Suite = Literal["formal", "reasoning", "compression"]
Kind = Literal["full", "lowres", "uniform", "static", "dynamic"]


@dataclass(frozen=True, slots=True)
class Condition:
    name: str
    kind: Kind
    budget_divisor: int
    use_top8: bool = False

    @property
    def native(self) -> bool:
        return self.kind in {"uniform", "static", "dynamic"}

    @property
    def routed(self) -> bool:
        return self.kind in {"static", "dynamic"}

    @property
    def lowres_divisor(self) -> int | None:
        return self.budget_divisor if self.kind == "lowres" else None

    @property
    def budget_area(self) -> int:
        return self.budget_divisor**2


FORMAL_CONDITIONS = (
    Condition("full", "full", 1),
    Condition("lowres4", "lowres", 4),
    Condition("uniform4_native", "uniform", 4),
    Condition("prefill_static_top8_native", "static", 4, use_top8=True),
    Condition("dynamic_top8_native", "dynamic", 4, use_top8=True),
    Condition("dynamic_all_heads_native", "dynamic", 4),
)

COMPRESSION_CONDITIONS = (
    Condition("lowres2", "lowres", 2),
    Condition("uniform2_native", "uniform", 2),
    Condition("prefill_static2_top8_native", "static", 2, use_top8=True),
    Condition("dynamic2_top8_native", "dynamic", 2, use_top8=True),
)

ALL_CONDITIONS = FORMAL_CONDITIONS + COMPRESSION_CONDITIONS
CONDITION_BY_NAME = {condition.name: condition for condition in ALL_CONDITIONS}


def conditions_for_suite(suite: Suite) -> tuple[Condition, ...]:
    if suite in {"formal", "reasoning"}:
        return FORMAL_CONDITIONS
    if suite == "compression":
        return COMPRESSION_CONDITIONS
    raise ValueError(f"unsupported E4 suite: {suite}")


def get_condition(name: str, suite: Suite | None = None) -> Condition:
    try:
        condition = CONDITION_BY_NAME[name]
    except KeyError as error:
        raise ValueError(
            f"unknown E4 condition {name!r}; choose from {sorted(CONDITION_BY_NAME)}"
        ) from error
    if suite is not None and condition not in conditions_for_suite(suite):
        raise ValueError(f"condition {name!r} is not part of E4 suite {suite!r}")
    return condition
