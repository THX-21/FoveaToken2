from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import torch

from experiments.e2.evaluator import load_tasks
from experiments.e2.runner import load_lm

from .conditions import CONDITIONS, Condition, get_condition
from .config import E3Config, as_dict
from .data import prepare_data
from .evaluator import PROMPT_VERSION, evaluate_condition


def run(
    config: E3Config,
    model_name: str,
    *,
    condition_name: str | None = None,
) -> Path:
    if model_name not in config.models:
        raise ValueError(f"unknown model {model_name!r}")
    prepare_data(config)
    requested = [get_condition(condition_name)] if condition_name else list(CONDITIONS)
    versions = _dependency_versions()
    output_dir = config.output_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    manifest = _manifest(config, model_name, requested, versions)
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["completed_conditions"] = list(previous.get("completed_conditions", []))
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    spec = config.models[model_name]
    lm = load_lm(spec, model_name)
    tasks = load_tasks(model_name, config.tasks)
    for condition in requested:
        evaluate_condition(config, model_name, spec, condition, lm, output_dir, tasks)
        if condition.name not in manifest["completed_conditions"]:
            manifest["completed_conditions"].append(condition.name)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return output_dir


def _manifest(
    config: E3Config,
    model_name: str,
    conditions: list[Condition],
    versions: dict[str, str],
) -> dict[str, Any]:
    return {
        "version": 2,
        "prompt_version": PROMPT_VERSION,
        "model_alias": model_name,
        "model": config.models[model_name].pretrained,
        "config": as_dict(config),
        "conditions": [condition.name for condition in conditions],
        "completed_conditions": [],
        "versions": versions,
    }


def _dependency_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "torch": torch.__version__}
    for package in ("transformers", "datasets", "lmms_eval"):
        try:
            module = __import__(package)
            versions[package] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[package] = "unavailable"
    return versions
