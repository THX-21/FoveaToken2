from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from experiments.local_datasets import load_lite_config

from .config import E2Config

DATASETS = {
    "vqav2_val_lite": ("vqav2_val", "lite"),
    "gqa_lite": ("gqa", "lite"),
    "textvqa_val_lite": ("textvqa_val", "lite"),
    "chartqa_lite": ("chartqa", "lite"),
}


def prepare_data(config: E2Config, *, force: bool = False) -> Path:
    manifest_path = config.data_dir / "sample_manifest.json"
    if manifest_path.exists() and not force:
        validate_manifest(config, json.loads(manifest_path.read_text(encoding="utf-8")))
        return manifest_path
    config.data_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"version": 1, "seed": config.seed, "tasks": {}}
    for task in config.tasks:
        if task not in DATASETS:
            raise ValueError(f"unsupported E2 task {task!r}")
        dataset_name, split = DATASETS[task]
        dataset = load_lite_config(dataset_name)
        if len(dataset) < config.sample_count:
            raise ValueError(f"{task} only has {len(dataset)} rows")
        rng = random.Random(f"{config.seed}:{task}")
        indices = sorted(rng.sample(range(len(dataset)), config.sample_count))
        payload["tasks"][task] = [
            {"sample_id": f"{task}:{index}", "source_index": index} for index in indices
        ]
    validate_manifest(config, payload)
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def validate_manifest(config: E2Config, payload: dict[str, Any]) -> None:
    if payload.get("seed") != config.seed:
        raise ValueError("E2 manifest seed does not match config")
    tasks = payload.get("tasks")
    if not isinstance(tasks, dict) or set(tasks) != set(config.tasks):
        raise ValueError("E2 manifest tasks do not match config")
    for task, records in tasks.items():
        if len(records) != config.sample_count:
            raise ValueError(f"{task} manifest must contain exactly {config.sample_count} samples")
        ids = [record["sample_id"] for record in records]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{task} manifest contains duplicate sample IDs")


def source_indices(config: E2Config, task: str) -> list[int]:
    payload = json.loads((config.data_dir / "sample_manifest.json").read_text(encoding="utf-8"))
    validate_manifest(config, payload)
    return [int(record["source_index"]) for record in payload["tasks"][task]]
