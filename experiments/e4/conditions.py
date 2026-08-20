from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

Suite = Literal["formal", "reasoning", "compression"]
Kind = Literal["full", "lowres", "uniform", "static", "dynamic"]


@dataclass(frozen=True, slots=True)
class Condition:
    name: str
    kind: Kind
    compression_ratio: float | None = None
    use_top8: bool = False

    @property
    def native(self) -> bool:
        return self.kind in {"uniform", "static", "dynamic"}

    @property
    def routed(self) -> bool:
        return self.kind in {"static", "dynamic"}


def ratio_label(compression_ratio: float) -> str:
    if not math.isfinite(compression_ratio) or compression_ratio <= 0:
        raise ValueError("compression ratio must be finite and positive")
    return f"{compression_ratio:g}".replace(".", "p")


def conditions_for_suite(
    suite: Suite,
    compression_ratio: float = 8.0,
    compression_ratios: tuple[float, ...] = (2.0, 4.0, 6.0, 8.0, 16.0),
) -> tuple[Condition, ...]:
    if suite in {"formal", "reasoning"}:
        return (
            Condition("full", "full"),
            *_ratio_conditions(compression_ratio),
            Condition(
                f"dynamic{ratio_label(compression_ratio)}_all_heads_native",
                "dynamic",
                compression_ratio,
            ),
        )
    if suite == "compression":
        return tuple(
            condition
            for ratio in compression_ratios
            for condition in _ratio_conditions(ratio)
        )
    raise ValueError(f"unsupported E4 suite: {suite}")


def _ratio_conditions(compression_ratio: float) -> tuple[Condition, ...]:
    label = ratio_label(compression_ratio)
    return (
        Condition(f"lowres{label}", "lowres", compression_ratio),
        Condition(f"uniform{label}_native", "uniform", compression_ratio),
        Condition(
            f"prefill_static{label}_top8_native",
            "static",
            compression_ratio,
            use_top8=True,
        ),
        Condition(
            f"dynamic{label}_top8_native",
            "dynamic",
            compression_ratio,
            use_top8=True,
        ),
    )


def get_condition(
    name: str,
    suite: Suite | None = None,
    compression_ratio: float = 8.0,
    compression_ratios: tuple[float, ...] = (2.0, 4.0, 6.0, 8.0, 16.0),
) -> Condition:
    suites: tuple[Suite, ...] = (
        (suite,) if suite is not None else ("formal", "compression")
    )
    available = {
        condition.name: condition
        for selected_suite in suites
        for condition in conditions_for_suite(
            selected_suite,
            compression_ratio,
            compression_ratios,
        )
    }
    try:
        return available[name]
    except KeyError as error:
        raise ValueError(
            f"unknown E4 condition {name!r}; choose from {sorted(available)}"
        ) from error
