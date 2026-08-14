from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Representation = Literal["full", "lowres2", "pool2", "native2"]
Position = Literal["mrope", "native_center", "text_anchor"]


@dataclass(frozen=True, slots=True)
class Condition:
    name: str
    representation: Representation
    position: Position

    @property
    def text_anchor(self) -> bool:
        return self.position == "text_anchor"

    @property
    def native(self) -> bool:
        return self.representation == "native2"

    @property
    def pooling(self) -> str:
        return "native_multiscale" if self.native else "kv"

    @property
    def compact(self) -> bool:
        return self.representation in {"pool2", "native2"}

    @property
    def block_size(self) -> int:
        return 2 if self.compact else 1


CONDITIONS = (
    Condition("full_mrope", "full", "mrope"),
    Condition("full_text_anchor", "full", "text_anchor"),
    Condition("lowres2_mrope", "lowres2", "mrope"),
    Condition("lowres2_text_anchor", "lowres2", "text_anchor"),
    Condition("pool2_center", "pool2", "native_center"),
    Condition("pool2_text_anchor", "pool2", "text_anchor"),
    Condition("native2_center", "native2", "native_center"),
    Condition("native2_text_anchor", "native2", "text_anchor"),
)

CONDITION_BY_NAME = {condition.name: condition for condition in CONDITIONS}

PAIRS = {
    "full_text_anchor": "full_mrope",
    "lowres2_text_anchor": "lowres2_mrope",
    "pool2_text_anchor": "pool2_center",
    "native2_text_anchor": "native2_center",
}


def get_condition(name: str) -> Condition:
    try:
        return CONDITION_BY_NAME[name]
    except KeyError as error:
        raise ValueError(
            f"unknown E3 condition {name!r}; choose from {sorted(CONDITION_BY_NAME)}"
        ) from error
