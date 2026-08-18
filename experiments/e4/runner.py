from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

import torch

from experiments.distributed import distributed_context

from .conditions import Condition, Suite, conditions_for_suite, get_condition
from .config import E4Config, as_dict
from .data import prepare_data
from .evaluator import evaluate_condition, load_tasks
from .runtime import validate_head_selection

RUN_PROTOCOL_VERSION = 1


def run(
    config: E4Config,
    model_name: str,
    suite: Suite,
    *,
    condition_name: str | None = None,
    task_name: str | None = None,
) -> Path:
    distributed = distributed_context()
    if model_name not in config.models:
        raise ValueError(f"unknown E4 model {model_name!r}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "E4 formal inference requires an NVIDIA CUDA environment; run prepare or smoke_e4 locally"
        )
    conditions = (
        [get_condition(condition_name, suite)]
        if condition_name
        else list(conditions_for_suite(suite))
    )
    if any(condition.use_top8 for condition in conditions):
        validate_head_selection(
            config.head_selections[model_name], config.models[model_name].pretrained
        )
    tasks = load_tasks(model_name, config.formal_tasks)
    if distributed.is_main:
        prepare_data(config, tasks)
    distributed.barrier()
    output_dir = config.output_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    manifest = _manifest(config, model_name)
    if distributed.is_main:
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous.get("version") != RUN_PROTOCOL_VERSION:
                raise ValueError("existing E4 outputs use an incompatible protocol version")
            manifest["completed"] = previous.get("completed", {})
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    distributed.barrier()
    lm = load_lm(config, model_name)
    for condition in conditions:
        result = evaluate_condition(
            config,
            model_name,
            condition,
            suite,
            lm,
            tasks,
            output_dir,
            task_name=task_name,
        )
        if distributed.is_main and result.get("status") != "partial":
            completed = manifest.setdefault("completed", {}).setdefault(suite, [])
            if condition.name not in completed:
                completed.append(condition.name)
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        distributed.barrier()
    return output_dir


def load_lm(config: E4Config, model_name: str) -> Any:
    spec = config.models[model_name]
    distributed = distributed_context()
    device = distributed.device if distributed.enabled else "cuda:0"
    if not torch.cuda.is_available():
        raise RuntimeError(
            "E4 formal inference requires an NVIDIA CUDA environment; run scripts/smoke_e4.py locally"
        )
    try:
        if model_name == "qwen25":
            from transformers import AutoProcessor
            from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL

            lm = Qwen2_5_VL(
                pretrained=spec.pretrained,
                batch_size=1,
                device=device,
                device_map=device,
                use_cache=True,
                attn_implementation="sdpa",
                min_pixels=spec.min_pixels,
                max_pixels=spec.max_pixels,
            )
            lm.processor = AutoProcessor.from_pretrained(
                spec.pretrained,
                min_pixels=spec.min_pixels,
                max_pixels=spec.max_pixels,
                use_fast=False,
            )
            return lm
        from lmms_eval.models.simple.qwen3_5 import Qwen3_5

        return Qwen3_5(
            pretrained=spec.pretrained,
            batch_size=1,
            device=device,
            device_map=device,
            use_cache=True,
            attn_implementation="sdpa",
            min_pixels=spec.min_pixels,
            max_pixels=spec.max_pixels,
            enable_thinking=False,
        )
    except (ImportError, RuntimeError) as error:
        raise RuntimeError(
            "E4 requires the NVIDIA environment and lmms-eval installation documented in its README"
        ) from error


def _manifest(config: E4Config, model_name: str) -> dict[str, Any]:
    versions = {"python": platform.python_version(), "torch": torch.__version__}
    for package in ("transformers", "datasets", "lmms_eval"):
        try:
            module = __import__(package)
            versions[package] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[package] = "unavailable"
    return {
        "version": RUN_PROTOCOL_VERSION,
        "model_alias": model_name,
        "model": config.models[model_name].pretrained,
        "config": as_dict(config),
        "thinking": False,
        "dtype": "bfloat16",
        "attention": "sdpa",
        "batch_size": 1,
        "seed": config.seed,
        "completed": {},
        "versions": versions,
    }
