from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import torch

from .conditions import CONDITIONS, Condition, get_condition
from .config import E2Config, ModelSpec, as_dict
from .data import prepare_data
from .evaluator import evaluate_condition, load_tasks

DEVICE = "cuda:0"
RUN_PROTOCOL_VERSION = 3


def run(
    config: E2Config,
    model_name: str,
    *,
    condition_name: str | None = None,
    thinking: bool = False,
) -> Path:
    if model_name not in config.models:
        raise ValueError(f"unknown model {model_name!r}")
    if thinking and model_name != "qwen35":
        raise ValueError("thinking is only supported for qwen35")
    prepare_data(config)
    spec = config.models[model_name]
    output_dir = config.output_dir / run_name(model_name, thinking)
    output_dir.mkdir(parents=True, exist_ok=True)
    conditions = [get_condition(condition_name)] if condition_name else list(CONDITIONS)
    manifest_path = output_dir / "run_manifest.json"
    manifest = _manifest(config, model_name, spec, conditions, thinking)
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("version") != RUN_PROTOCOL_VERSION:
            raise ValueError(
                f"{output_dir} uses E2 result protocol {previous.get('version')!r}; "
                "move or remove it before rerunning with the corrected task prompts"
            )
        manifest["completed_conditions"] = list(previous.get("completed_conditions", []))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    lm = load_lm(spec, model_name, thinking=thinking)
    tasks = load_tasks(model_name, config.tasks)
    for condition in conditions:
        evaluate_condition(config, model_name, spec, condition, lm, output_dir, tasks, thinking=thinking)
        if condition.name not in manifest["completed_conditions"]:
            manifest["completed_conditions"].append(condition.name)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_dir


def run_name(model_name: str, thinking: bool = False) -> str:
    return f"{model_name}_thinking" if thinking else model_name


def load_lm(spec: ModelSpec, model_name: str, *, thinking: bool = False) -> Any:
    try:
        if model_name == "qwen25":
            from transformers import AutoProcessor
            from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL
            cls = Qwen2_5_VL
            extra = {}
        else:
            from lmms_eval.models.simple.qwen3_5 import Qwen3_5
            cls = Qwen3_5
            extra = {"enable_thinking": thinking}
        lm = cls(
            pretrained=spec.pretrained,
            batch_size=1,
            device=DEVICE,
            device_map=DEVICE,
            use_cache=True,
            attn_implementation="sdpa",
            min_pixels=spec.min_pixels,
            max_pixels=spec.max_pixels,
            **extra,
        )
        if model_name == "qwen25":
            lm.processor = AutoProcessor.from_pretrained(
                spec.pretrained,
                min_pixels=spec.min_pixels,
                max_pixels=spec.max_pixels,
                use_fast=False,
            )
        return lm
    except (ImportError, RuntimeError) as error:
        raise RuntimeError(
            "E2 model execution requires the NVIDIA environment documented in experiments/e2/README.md"
        ) from error


def _manifest(
    config: E2Config,
    model_name: str,
    spec: ModelSpec,
    conditions: list[Condition],
    thinking: bool,
) -> dict[str, Any]:
    versions: dict[str, str] = {"python": platform.python_version(), "torch": torch.__version__}
    for package in ("transformers", "datasets", "lmms_eval"):
        try:
            module = __import__(package)
            versions[package] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[package] = "unavailable"
    return {
        "version": RUN_PROTOCOL_VERSION,
        "model_alias": model_name,
        "thinking": thinking,
        "max_new_tokens": 2048 if thinking else config.max_new_tokens,
        "model": spec.pretrained,
        "config": as_dict(config),
        "conditions": [condition.name for condition in conditions],
        "completed_conditions": [],
        "versions": versions,
    }
