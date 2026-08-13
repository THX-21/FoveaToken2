from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

DEFAULT_TASKS = ("vqav2_val_lite", "gqa_lite", "textvqa_val_lite", "chartqa_lite")
DEFAULT_AREA_RATIOS = (0.50, 0.30, 0.20)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    pretrained: str
    min_pixels: int
    max_pixels: int
    pixel_per_token: int


@dataclass(slots=True)
class E2Config:
    seed: int = 42
    sample_count: int = 100
    max_new_tokens: int = 16
    data_dir: Path = Path("data/e2")
    output_dir: Path = Path("outputs/e2")
    tasks: tuple[str, ...] = DEFAULT_TASKS
    area_ratios: tuple[float, float, float] = DEFAULT_AREA_RATIOS
    models: dict[str, ModelSpec] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "E2Config":
        config_path = Path(path)
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        models = {name: ModelSpec(**values) for name, values in payload.pop("models", {}).items()}
        payload["models"] = models
        payload["tasks"] = tuple(payload.get("tasks", DEFAULT_TASKS))
        payload["area_ratios"] = tuple(payload.get("area_ratios", DEFAULT_AREA_RATIOS))
        payload["data_dir"] = Path(payload.get("data_dir", "data/e2")).expanduser()
        payload["output_dir"] = Path(payload.get("output_dir", "outputs/e2")).expanduser()
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        if self.seed < 0 or self.sample_count <= 0 or self.max_new_tokens <= 0:
            raise ValueError("seed must be non-negative and counts must be positive")
        if len(self.area_ratios) != 3 or abs(sum(self.area_ratios) - 1.0) > 1e-9:
            raise ValueError("area_ratios must contain three values summing to one")
        if not self.tasks or not self.models:
            raise ValueError("E2 requires tasks and model specifications")
        for spec in self.models.values():
            if spec.pixel_per_token <= 0 or spec.min_pixels <= 0 or spec.max_pixels < spec.min_pixels:
                raise ValueError("invalid model pixel limits")


def as_dict(config: E2Config) -> dict[str, Any]:
    return {
        "seed": config.seed,
        "sample_count": config.sample_count,
        "max_new_tokens": config.max_new_tokens,
        "tasks": list(config.tasks),
        "area_ratios": list(config.area_ratios),
    }
