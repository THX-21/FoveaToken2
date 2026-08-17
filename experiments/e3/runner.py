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
from .evaluator import (
    PROMPT_VERSION,
    SCORING_VERSION,
    evaluate_condition,
    llm_judge_status,
    reevaluate_condition,
)

RUN_PROTOCOL_VERSION = 2


def run(
    config: E3Config,
    model_name: str,
    *,
    condition_name: str | None = None,
    scoring_version: str = SCORING_VERSION,
) -> Path:
    if model_name not in config.models:
        raise ValueError(f"unknown model {model_name!r}")
    judge_enabled, judge_model, judge_max_tokens, judge_thinking = llm_judge_status()
    if judge_enabled:
        print(
            "E3 LLM Judge: enabled "
            f"(model={judge_model}, max_tokens={judge_max_tokens}, thinking={judge_thinking}; "
            "triggers on invalid format or rule-score mismatch)."
        )
    else:
        print("E3 LLM Judge: disabled.")
    prepare_data(config)
    requested = [get_condition(condition_name)] if condition_name else list(CONDITIONS)
    versions = _dependency_versions()
    output_dir = config.output_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    manifest = _manifest(config, model_name, requested, versions)
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("version") != RUN_PROTOCOL_VERSION:
            raise ValueError(
                f"{output_dir} uses E3 result protocol {previous.get('version')!r}; "
                "move or remove it before rerunning"
            )
        if previous.get("prompt_version") != PROMPT_VERSION:
            raise ValueError(
                f"{output_dir} uses E3 prompt version {previous.get('prompt_version')!r}; "
                "move or remove it before rerunning"
            )
        manifest["completed_conditions"] = list(previous.get("completed_conditions", []))
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    spec = config.models[model_name]
    lm = load_lm(spec, model_name)
    tasks = load_tasks(model_name, config.tasks)
    for condition in requested:
        evaluate_condition(
            config,
            model_name,
            spec,
            condition,
            lm,
            output_dir,
            tasks,
            scoring_version=scoring_version,
        )
        if condition.name not in manifest["completed_conditions"]:
            manifest["completed_conditions"].append(condition.name)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return output_dir


def reevaluate(
    config: E3Config,
    model_name: str,
    *,
    condition_name: str | None = None,
    scoring_version: str = SCORING_VERSION,
    restart: bool = False,
    workers: int = 8,
) -> Path:
    if model_name not in config.models:
        raise ValueError(f"unknown model {model_name!r}")
    if workers <= 0:
        raise ValueError("workers must be positive")
    judge_enabled, judge_model, judge_max_tokens, judge_thinking = llm_judge_status()
    print(
        f"E3 LLM Judge: {'enabled' if judge_enabled else 'disabled'}"
        + (
            f" (model={judge_model}, max_tokens={judge_max_tokens}, thinking={judge_thinking})."
            if judge_enabled
            else "."
        )
    )
    print(
        f"E3 reevaluate: workers={workers}, version={scoring_version}, "
        f"mode={'restart' if restart else 'resume'}."
    )
    prepare_data(config)
    requested = [get_condition(condition_name)] if condition_name else list(CONDITIONS)
    output_dir = config.output_dir / model_name
    tasks = load_tasks(model_name, config.tasks)
    for condition in requested:
        payload = reevaluate_condition(
            config,
            model_name,
            config.models[model_name],
            condition,
            output_dir,
            tasks,
            scoring_version=scoring_version,
            restart=restart,
            workers=workers,
        )
        print(f"reevaluated {condition.name}: macro_average={payload['macro_average']:.4f}")
    return output_dir


def _manifest(
    config: E3Config,
    model_name: str,
    conditions: list[Condition],
    versions: dict[str, str],
) -> dict[str, Any]:
    return {
        "version": RUN_PROTOCOL_VERSION,
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
