from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

FORMAL_TASKS = (
    "hrbench8k",
    "xlrs-lite",
    "vstar_bench",
    "visualprobe_easy",
    "visualprobe_medium",
    "visualprobe_hard",
    "finers_qa",
    "hrscene_testmini",
    "mmstar",
    "chartqa",
    "textvqa_val",
)

MECHANISM_TASKS = (
    "vstar_bench",
    "visualprobe_easy",
    "visualprobe_medium",
    "visualprobe_hard",
    "finers_qa",
    "hrscene_testmini",
)

PRIMARY_METRICS = {
    "hrbench8k": "average",
    "xlrs-lite": "xlrs_micro_score",
    "vstar_bench": "vstar_overall_acc",
    "visualprobe_easy": "visualprobe_accuracy",
    "visualprobe_medium": "visualprobe_accuracy",
    "visualprobe_hard": "visualprobe_accuracy",
    "finers_qa": "finers_qa_accuracy",
    "hrscene_testmini": "hrscene_accuracy",
    "mmstar": "average",
    "chartqa": "relaxed_overall",
    "textvqa_val": "exact_match",
}


@dataclass(frozen=True, slots=True)
class ModelSpec:
    pretrained: str
    min_pixels: int
    max_pixels: int
    pixel_per_token: int


@dataclass(slots=True)
class E4Config:
    seed: int = 42
    visual_token_cap: int = 4096
    compression_ratio: float = 8.0
    compression_ratios: tuple[float, ...] = (2.0, 4.0, 6.0, 8.0, 16.0)
    formal_max_new_tokens: int = 128
    reasoning_max_new_tokens: int = 256
    mechanism_count: int = 100
    data_dir: Path = Path("data/e4")
    output_dir: Path = Path("outputs/e4")
    formal_tasks: tuple[str, ...] = FORMAL_TASKS
    mechanism_tasks: tuple[str, ...] = MECHANISM_TASKS
    primary_metrics: dict[str, str] = field(default_factory=lambda: dict(PRIMARY_METRICS))
    head_selections: dict[str, Path] = field(default_factory=dict)
    models: dict[str, ModelSpec] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "E4Config":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        payload["models"] = {
            name: ModelSpec(**values) for name, values in payload.get("models", {}).items()
        }
        payload["compression_ratios"] = tuple(
            float(value)
            for value in payload.get("compression_ratios", (2, 4, 6, 8, 16))
        )
        payload["formal_tasks"] = tuple(payload.get("formal_tasks", FORMAL_TASKS))
        payload["mechanism_tasks"] = tuple(payload.get("mechanism_tasks", MECHANISM_TASKS))
        payload["primary_metrics"] = {
            **PRIMARY_METRICS,
            **payload.get("primary_metrics", {}),
        }
        payload["head_selections"] = {
            name: Path(value).expanduser()
            for name, value in payload.get("head_selections", {}).items()
        }
        payload["data_dir"] = Path(payload.get("data_dir", "data/e4")).expanduser()
        payload["output_dir"] = Path(payload.get("output_dir", "outputs/e4")).expanduser()
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.visual_token_cap <= 0 or self.visual_token_cap % 64:
            raise ValueError("visual_token_cap must be positive and divisible by 64")
        if (
            not math.isfinite(self.compression_ratio)
            or not 1.0 <= self.compression_ratio <= 64.0
        ):
            raise ValueError("compression_ratio must be finite and between 1 and 64")
        if (
            not self.compression_ratios
            or len(set(self.compression_ratios)) != len(self.compression_ratios)
            or any(
                not math.isfinite(value) or not 1.0 <= value <= 64.0
                for value in self.compression_ratios
            )
        ):
            raise ValueError(
                "compression_ratios must contain unique values between 1 and 64"
            )
        if min(self.formal_max_new_tokens, self.reasoning_max_new_tokens, self.mechanism_count) <= 0:
            raise ValueError("generation and mechanism counts must be positive")
        if not self.formal_tasks or not self.mechanism_tasks or not self.models:
            raise ValueError("E4 requires tasks and model specifications")
        missing = set(self.formal_tasks) - set(self.primary_metrics)
        if missing:
            raise ValueError(f"missing E4 primary metrics for tasks: {sorted(missing)}")
        if set(self.head_selections) != set(self.models):
            raise ValueError("E4 head_selections must contain exactly one path per model")
        for model_name, spec in self.models.items():
            if spec.pixel_per_token <= 0 or spec.min_pixels <= 0:
                raise ValueError(f"invalid E4 model specification for {model_name}")
            expected = self.visual_token_cap * spec.pixel_per_token**2
            if spec.max_pixels != expected:
                raise ValueError(
                    f"{model_name} max_pixels must equal visual_token_cap * pixel_per_token^2 ({expected})"
                )


def as_dict(config: E4Config) -> dict[str, Any]:
    return {
        "seed": config.seed,
        "visual_token_cap": config.visual_token_cap,
        "compression_ratio": config.compression_ratio,
        "compression_ratios": list(config.compression_ratios),
        "formal_max_new_tokens": config.formal_max_new_tokens,
        "reasoning_max_new_tokens": config.reasoning_max_new_tokens,
        "mechanism_count": config.mechanism_count,
        "formal_tasks": list(config.formal_tasks),
        "mechanism_tasks": list(config.mechanism_tasks),
        "primary_metrics": dict(config.primary_metrics),
        "head_selections": {name: str(path) for name, path in config.head_selections.items()},
    }
