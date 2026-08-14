from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(slots=True)
class FoveaConfig:
    """Core fixed-budget routing configuration."""

    budget: int = 1024
    mode: Literal["dynamic", "uniform", "full"] = "dynamic"
    position_mode: Literal["native_center", "text_anchor", "no_rope", "post_rope_pool"] = "native_center"
    pooling_mode: Literal["kv", "hidden", "native_multiscale"] = "kv"
    signal_selection: str | None = None
    signal_aggregation: Literal["mean", "max"] = "mean"
    anchor_window: float = 8.0
    update_interval: int = 1
    max_swaps: int = 100
    epsilon: float = 0.05
    attention_ema: float = 0.0
    score_mode: Literal["mass", "density"] = "mass"
    route_after_prefill: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"dynamic", "uniform", "full"}:
            raise ValueError(f"unsupported mode: {self.mode}")
        if self.position_mode not in {"native_center", "text_anchor", "no_rope", "post_rope_pool"}:
            raise ValueError(f"unsupported position_mode: {self.position_mode}")
        if self.pooling_mode not in {"kv", "hidden", "native_multiscale"}:
            raise ValueError(f"unsupported pooling_mode: {self.pooling_mode}")
        if self.signal_aggregation not in {"mean", "max"}:
            raise ValueError(f"unsupported signal_aggregation: {self.signal_aggregation}")
        if self.pooling_mode == "hidden" and self.position_mode == "post_rope_pool":
            raise ValueError("hidden pooling cannot be combined with post_rope_pool")
        if self.pooling_mode == "native_multiscale" and self.position_mode == "post_rope_pool":
            raise ValueError("native_multiscale cannot be combined with post_rope_pool")
        if self.anchor_window <= 0:
            raise ValueError("anchor_window must be positive")
        if self.budget <= 0:
            raise ValueError("budget must be positive")
        if self.update_interval <= 0:
            raise ValueError("update_interval must be positive")
        if self.max_swaps < 0:
            raise ValueError("max_swaps must be non-negative")
        if self.epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        if not 0.0 <= self.attention_ema < 1.0:
            raise ValueError("attention_ema must be in [0, 1)")
        if self.score_mode not in {"mass", "density"}:
            raise ValueError(f"unsupported score_mode: {self.score_mode}")
