from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm  # type: ignore[import-untyped]

from experiments.distributed import (
    distributed_context,
    merge_rank_jsonl,
    merge_rank_text,
)
from experiments.local_datasets import use_local_lmms_datasets
from tokenfovea.generation import generate_with_prefill_boundary

from .conditions import Condition
from .config import E2Config, ModelSpec
from .data import source_indices
from .front import BlockFront, stable_seed
from .image import aligned_high_resolution, lowres_plan, matched_lowres_plan, resize
from .patch import install_e2
from .session import E2Session

RANDOM_PERSTEP_PREFILL_PROTOCOL = "compact_prefix_boundary_v1"


def evaluate_condition(
    config: E2Config,
    model_name: str,
    spec: ModelSpec,
    condition: Condition,
    lm: Any,
    output_dir: Path,
    tasks: dict[str, Any] | None = None,
    *,
    thinking: bool = False,
) -> dict[str, Any]:
    distributed = distributed_context()
    condition_dir = output_dir / condition.name
    condition_dir.mkdir(parents=True, exist_ok=True)
    result_path = condition_dir / "results.json"
    sample_path = condition_dir / "samples.jsonl"
    trace_path = condition_dir / "front_traces.jsonl"
    if distributed.is_main:
        merge_rank_jsonl(sample_path, key="sample_id")
        merge_rank_text(trace_path)
    distributed.barrier()
    task_indices = {task_name: source_indices(config, task_name) for task_name in config.tasks}
    expected_sample_ids = {
        f"{task_name}:{source_index}"
        for task_name, indices in task_indices.items()
        for source_index in indices
    }
    completed_samples = _read_completed_samples(
        sample_path,
        expected_sample_ids,
        required_prefill_protocol=(
            RANDOM_PERSTEP_PREFILL_PROTOCOL
            if condition.front_mode == "random_perstep"
            else None
        ),
    )
    if result_path.exists() and len(completed_samples) == len(expected_sample_ids):
        return json.loads(result_path.read_text(encoding="utf-8"))
    if distributed.is_main and result_path.exists():
        result_path.unlink()
    distributed.barrier()
    if distributed.is_main and trace_path.exists() and not completed_samples:
        trace_path.unlink()
    distributed.barrier()
    write_sample_path = (
        distributed.rank_path(sample_path) if distributed.enabled else sample_path
    )
    write_trace_path = (
        distributed.rank_path(trace_path) if distributed.enabled else trace_path
    )
    session = E2Session(
        condition,
        seed=config.seed,
        area_ratios=config.area_ratios,
        trace_path=write_trace_path if condition.pooled else None,
    )
    patch = install_e2(lm.model, session) if _needs_patch(condition) else None
    try:
        task_manager = _task_manager(model_name) if tasks is None else None
        for task_name in config.tasks:
            task = tasks[task_name] if tasks is not None else _load_task(task_manager, task_name)
            documents = task.eval_docs
            indices = task_indices[task_name]
            sample_ids = [f"{task_name}:{source_index}" for source_index in indices]
            pending_indices = [
                source_index
                for source_index, sample_id in zip(indices, sample_ids)
                if sample_id not in completed_samples
            ]
            local_indices = distributed.shard(pending_indices)
            for source_index in tqdm(
                local_indices,
                total=len(local_indices),
                desc=(
                    f"{model_name} {condition.name} {task_name} "
                    f"rank {distributed.rank}/{distributed.world_size}"
                ),
            ):
                doc = documents[source_index]
                sample_id = f"{task_name}:{source_index}"
                prompt = task.doc_to_text(doc)
                visuals = task.doc_to_visual(doc)
                if not isinstance(visuals, list) or len(visuals) != 1 or not isinstance(visuals[0], Image.Image):
                    raise ValueError("E2 requires one PIL image per sample")
                answer, record = _generate(
                    config, model_name, spec, condition, lm, session,
                    sample_id, visuals[0], str(prompt), task.get_config("generation_kwargs") or {}, thinking,
                )
                processed = task.process_results(doc, [answer])
                record.update(
                    {
                        "task": task_name,
                        "source_index": source_index,
                        "prediction": answer,
                        "target": task.doc_to_target(doc),
                        "metrics": _jsonable(processed),
                    }
                )
                completed_samples[sample_id] = record
                with write_sample_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        distributed.barrier()
        if distributed.is_main:
            merge_rank_jsonl(sample_path, key="sample_id")
            merge_rank_text(trace_path)
            completed = _read_completed_samples(
                sample_path,
                expected_sample_ids,
                required_prefill_protocol=(
                    RANDOM_PERSTEP_PREFILL_PROTOCOL
                    if condition.front_mode == "random_perstep"
                    else None
                ),
            )
            payload = _aggregate_condition(
                config,
                model_name,
                spec,
                condition,
                tasks,
                task_manager,
                task_indices,
                completed,
            )
            result_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        distributed.barrier()
        return json.loads(result_path.read_text(encoding="utf-8"))
    finally:
        if patch is not None:
            patch.remove()


def _aggregate_condition(
    config: E2Config,
    model_name: str,
    spec: ModelSpec,
    condition: Condition,
    tasks: dict[str, Any] | None,
    task_manager: Any,
    task_indices: dict[str, list[int]],
    completed: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    task_results: dict[str, Any] = {}
    for task_name in config.tasks:
        task = tasks[task_name] if tasks is not None else _load_task(task_manager, task_name)
        sample_ids = [f"{task_name}:{index}" for index in task_indices[task_name]]
        missing = [sample_id for sample_id in sample_ids if sample_id not in completed]
        if missing:
            raise RuntimeError(
                f"distributed E2 condition {condition.name} is missing {len(missing)} samples"
            )
        rows = [completed[sample_id] for sample_id in sample_ids]
        metric_values: dict[str, list[Any]] = {}
        for record in rows:
            for name, value in record["metrics"].items():
                if isinstance(value, (int, float, bool)):
                    metric_values.setdefault(name, []).append(value)
        aggregators = task.aggregation()
        metrics = {
            name: float(
                aggregators[name](values)
                if aggregators.get(name) is not None
                else sum(values) / len(values)
            )
            for name, values in metric_values.items()
        }
        task_results[task_name] = {
            "metrics": metrics,
            "primary_score": _primary_metric(task_name, metrics),
            "samples": len(rows),
        }
    return {
        "model": spec.pretrained,
        "model_alias": model_name,
        "condition": condition.name,
        "tasks": task_results,
        "macro_average": sum(row["primary_score"] for row in task_results.values())
        / len(task_results),
    }


def _read_completed_samples(
    path: Path,
    expected_ids: set[str],
    *,
    required_prefill_protocol: str | None = None,
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    completed: dict[str, dict[str, Any]] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break
            raise
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id not in expected_ids:
            raise ValueError(f"invalid E2 sample ID in {path}: {sample_id!r}")
        if sample_id in completed:
            raise ValueError(f"duplicate E2 sample ID in {path}: {sample_id!r}")
        if not isinstance(row.get("metrics"), dict):
            raise ValueError(f"missing E2 metrics in {path}: {sample_id!r}")
        if (
            required_prefill_protocol is not None
            and row.get("prefill_protocol") != required_prefill_protocol
        ):
            raise ValueError(
                f"obsolete E2 random-per-step prefill protocol in {path}; "
                "move or remove this condition directory before rerunning"
            )
        completed[sample_id] = row
    return completed


def _needs_patch(condition: Condition) -> bool:
    return condition.pooled


def _generate(
    config: E2Config,
    model_name: str,
    spec: ModelSpec,
    condition: Condition,
    lm: Any,
    session: E2Session,
    sample_id: str,
    image: Image.Image,
    prompt: str,
    generation_kwargs: dict[str, Any],
    thinking: bool = False,
) -> tuple[str, dict[str, Any]]:
    high = aligned_high_resolution(image, spec.pixel_per_token, spec.min_pixels, spec.max_pixels)
    target_tokens = None
    if condition.lowres_divisor:
        plan = lowres_plan(high, condition.lowres_divisor)
        target_tokens = high.visual_tokens // (condition.lowres_divisor**2)
    elif condition.name == "lowres_random_matched":
        front = BlockFront.random_multiscale(
            high.grid_height, high.grid_width,
            stable_seed(config.seed, sample_id), config.area_ratios,
        )
        target_tokens = front.node_count
        plan = matched_lowres_plan(high, target_tokens, spec.pixel_per_token)
    else:
        plan = high
    prepared_image = resize(image, plan)
    session.begin_sample(sample_id)
    if condition.native:
        try:
            for divisor, area_scale in ((2, 4), (4, 16)):
                auxiliary_plan = lowres_plan(high, divisor)
                auxiliary_inputs = _prepare_inputs(
                    lm,
                    model_name,
                    resize(image, auxiliary_plan),
                    prompt,
                    auxiliary_plan,
                    thinking,
                )
                session.begin_native_capture(area_scale)
                with torch.inference_mode():
                    auxiliary_output = lm.model(
                        **auxiliary_inputs,
                        use_cache=True,
                        return_dict=True,
                    )
                del auxiliary_output
                session.end_native_capture()
        except Exception:
            session.abort_native_sample()
            raise
    inputs = _prepare_inputs(lm, model_name, prepared_image, prompt, plan, thinking)
    kwargs = {
        "max_new_tokens": _max_new_tokens(config, generation_kwargs, thinking),
        "do_sample": False,
        "num_beams": 1,
        "use_cache": True,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = (
            generate_with_prefill_boundary(lm.model, inputs, kwargs)
            if condition.front_mode == "random_perstep"
            else lm.model.generate(**inputs, **kwargs)
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    prompt_length = inputs["input_ids"].shape[-1]
    generated = output.sequences[0, prompt_length:]
    answer = lm.processor.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    until = generation_kwargs.get("until", [])
    if isinstance(until, str):
        until = [until]
    for term in until:
        if term:
            answer = answer.split(term)[0]
    answer = answer.strip()
    score_tokens = [int(scores[0].argmax()) for scores in output.scores]
    diagnostics = session.diagnostics()
    diagnostics.update(
        {
            "input_size": [plan.height, plan.width],
            "grid": [plan.grid_height, plan.grid_width],
            "visual_tokens": plan.visual_tokens,
            "highres_visual_tokens": high.visual_tokens,
            "target_visual_tokens": target_tokens,
            "token_count_delta": plan.visual_tokens - target_tokens if target_tokens is not None else 0,
            "generated_token_ids": generated.detach().cpu().tolist(),
            "score_top1_token_ids": score_tokens,
            "total_seconds": elapsed,
        }
    )
    if condition.front_mode == "random_perstep":
        diagnostics["prefill_protocol"] = RANDOM_PERSTEP_PREFILL_PROTOCOL
    if diagnostics["active_tokens"] is None:
        diagnostics["active_tokens"] = plan.visual_tokens
        diagnostics["compression_ratio"] = plan.visual_tokens / high.visual_tokens
    return answer, diagnostics


def _prepare_inputs(
    lm: Any,
    model_name: str,
    image: Image.Image,
    prompt: str,
    plan: Any,
    thinking: bool = False,
):
    messages = _messages(lm, image, prompt)
    text = _chat_template(lm, messages, model_name, thinking)
    inputs = lm.processor(
        text=[text], images=[image], do_resize=False, return_tensors="pt"
    ).to(lm.device)
    actual_grid = tuple(int(value) for value in inputs["image_grid_thw"][0].tolist())
    merge_size = int(lm.model.config.vision_config.spatial_merge_size)
    expected_grid = (1, plan.grid_height * merge_size, plan.grid_width * merge_size)
    if actual_grid != expected_grid:
        raise ValueError(f"processor grid {actual_grid} does not match E2 plan {expected_grid}")
    _validate_image_token_count(inputs, lm.model)
    return inputs


def _task_manager(model_name: str):
    from lmms_eval.tasks import TaskManager
    return TaskManager(verbosity="ERROR", model_name=_lmms_model_name(model_name))


def _lmms_model_name(model_name: str) -> str:
    return "qwen2_5_vl" if model_name == "qwen25" else "qwen3_5"


def _max_new_tokens(config: E2Config, generation_kwargs: dict[str, Any], thinking: bool) -> int:
    if thinking:
        return 2048
    return min(int(generation_kwargs.get("max_new_tokens", config.max_new_tokens)), config.max_new_tokens)


def _validate_image_token_count(inputs: Any, model: Any) -> None:
    image_tokens = int((inputs["input_ids"] == model.config.image_token_id).sum())
    expected_tokens = int(inputs["image_grid_thw"].prod()) // int(
        model.config.vision_config.spatial_merge_size
    ) ** 2
    if image_tokens != expected_tokens:
        raise ValueError(
            f"E2 image token count {image_tokens} does not match visual features {expected_tokens}"
        )


def _load_task(manager: Any, task_name: str) -> Any:
    from lmms_eval.tasks import get_task_dict
    task = get_task_dict([task_name], manager, "simple")[task_name]
    return task[1] if isinstance(task, tuple) else task


def load_tasks(model_name: str, task_names: tuple[str, ...]) -> dict[str, Any]:
    with use_local_lmms_datasets():
        manager = _task_manager(model_name)
        return {name: _load_task(manager, name) for name in task_names}


def _messages(lm: Any, image: Image.Image, prompt: str) -> list[dict[str, Any]]:
    messages = []
    if getattr(lm, "system_prompt", None):
        messages.append({"role": "system", "content": lm.system_prompt})
    messages.append(
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}
    )
    return messages


def _chat_template(lm: Any, messages: list[dict[str, Any]], model_name: str, thinking: bool = False) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if model_name == "qwen35":
        kwargs["enable_thinking"] = thinking
    return lm.processor.apply_chat_template(messages, **kwargs)


def _primary_metric(task: str, metrics: dict[str, float]) -> float:
    name = "relaxed_overall" if task == "chartqa_lite" else "exact_match"
    if name not in metrics:
        raise ValueError(f"{task} did not produce primary metric {name}")
    return metrics[name]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
