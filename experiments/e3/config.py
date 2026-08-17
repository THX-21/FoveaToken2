from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from experiments.e2.config import DEFAULT_AREA_RATIOS, DEFAULT_TASKS, E2Config, ModelSpec


@dataclass(slots=True)
class E3Config:
    seed: int = 42
    sample_count: int = 100
    max_new_tokens: int = 1024
    anchor_window: float = 2.0
    e2_data_dir: Path = Path("data/e2")
    output_dir: Path = Path("outputs/e3")
    tasks: tuple[str, ...] = DEFAULT_TASKS
    models: dict[str, ModelSpec] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "E3Config":
        config_path = Path(path)
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        payload["models"] = {
            name: ModelSpec(**values) for name, values in payload.get("models", {}).items()
        }
        payload["tasks"] = tuple(payload.get("tasks", DEFAULT_TASKS))
        for name, default in (
            ("e2_data_dir", "data/e2"),
            ("output_dir", "outputs/e3"),
        ):
            payload[name] = Path(payload.get(name, default)).expanduser()
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        if self.seed < 0 or self.sample_count <= 0 or self.max_new_tokens <= 0:
            raise ValueError("seed must be non-negative and counts must be positive")
        if self.anchor_window < 0:
            raise ValueError("anchor_window must be non-negative")
        if not self.tasks or not self.models:
            raise ValueError("E3 requires tasks and model specifications")
        for spec in self.models.values():
            if spec.pixel_per_token <= 0 or spec.min_pixels <= 0 or spec.max_pixels < spec.min_pixels:
                raise ValueError("invalid model pixel limits")

    def e2_config(self) -> E2Config:
        config = E2Config(
            seed=self.seed,
            sample_count=self.sample_count,
            max_new_tokens=self.max_new_tokens,
            data_dir=self.e2_data_dir,
            output_dir=Path("outputs/e2"),
            tasks=self.tasks,
            area_ratios=DEFAULT_AREA_RATIOS,
            models=self.models,
        )
        config.validate()
        return config


def as_dict(config: E3Config) -> dict[str, Any]:
    return {
        "seed": config.seed,
        "sample_count": config.sample_count,
        "max_new_tokens": config.max_new_tokens,
        "anchor_window": config.anchor_window,
        "tasks": list(config.tasks),
    }
