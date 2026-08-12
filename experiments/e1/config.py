from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


@dataclass(slots=True)
class NaturalSource:
    name: str
    dataset_name: str
    count: int = 100
    prompt: str = "Provide a concise one-sentence description of this image."


@dataclass(slots=True)
class ModelSpec:
    pretrained: str
    min_pixels: int
    max_pixels: int


@dataclass(slots=True)
class E1Config:
    seed: int = 42
    data_dir: Path = Path("data/e1")
    output_dir: Path = Path("outputs/e1")
    natural_sources: list[NaturalSource] = field(default_factory=list)
    controlled_count: int = 100
    generation_tokens: int = 32
    basic_keep_fraction: float = 0.20
    hybrid_top_fraction: float = 0.05
    visualization_natural: int = 12
    visualization_controlled: int = 6
    models: dict[str, ModelSpec] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "E1Config":
        config_path = Path(path)
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        natural = [NaturalSource(**item) for item in payload.get("natural_sources", [])]
        models = {name: ModelSpec(**values) for name, values in payload.get("models", {}).items()}
        values: dict[str, Any] = {
            key: value
            for key, value in payload.items()
            if key not in {"natural_sources", "models"}
        }
        values["natural_sources"] = natural
        values["models"] = models
        values["data_dir"] = _relative_path(config_path, values.get("data_dir", "data/e1"))
        values["output_dir"] = _relative_path(config_path, values.get("output_dir", "outputs/e1"))
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        if not self.natural_sources:
            raise ValueError("E1 requires at least one natural image source")
        if not self.models:
            raise ValueError("E1 requires at least one model specification")
        if not 0.0 < self.basic_keep_fraction <= 1.0:
            raise ValueError("basic_keep_fraction must be in (0, 1]")
        if not 0.0 < self.hybrid_top_fraction <= 1.0:
            raise ValueError("hybrid_top_fraction must be in (0, 1]")
        if self.controlled_count <= 0 or self.generation_tokens <= 0:
            raise ValueError("controlled_count and generation_tokens must be positive")


def _relative_path(config_path: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    # Repository-facing defaults remain relative to the current working directory.
    if not str(path).startswith("."):
        return path
    return (config_path.parent / path).resolve()
