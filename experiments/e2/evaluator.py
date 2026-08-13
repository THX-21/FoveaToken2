from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm  # type: ignore[import-untyped]

from .conditions import Condition
from .config import E2Config, ModelSpec
from .data import source_indices
from .front import BlockFront, stable_seed
from .image import aligned_high_resolution, lowres_plan, matched_lowres_plan, resize
from .patch import install_e2
from .session import E2Session


def evaluate_condition(
    config: E2Config,
    model_name: str,
    spec: ModelSpec,
    condition: Condition,
    lm: Any,
    output_dir: Path,
    tasks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    condition_dir = output_dir / condition.name
    condition_dir.mkdir(parents=True, exist_ok=True)
    result_path = condition_dir / "results.json"
    sample_path = condition_dir / "samples.jsonl"
    if result_path.exists() and sample_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    if sample_path.exists():
        sample_path.unlink()
    trace_path = condition_dir / "front_traces.jsonl"
    if trace_path.exists():
        trace_path.unlink()
    session = E2Session(
        condition,
        seed=config.seed,
        area_ratios=config.area_ratios,
        trace_path=trace_path if condition.pooled else None,
    )
    # Full and LowRes use the same hook only for timing; their attention forward returns unchanged.
    patch = install_e2(lm.model, session)
    try:
        task_manager = _task_manager(model_name) if tasks is None else None
        task_results: dict[str, Any] = {}
        for task_name in config.tasks:
            task = tasks[task_name] if tasks is not None else _load_task(task_manager, task_name)
            documents = task.eval_docs
            rows: list[dict[str, Any]] = []
            metric_values: dict[str, list[Any]] = {}
            for source_index in tqdm(
                source_indices(config, task_name), desc=f"{model_name} {condition.name} {task_name}"
            ):
                doc = documents[source_index]
                sample_id = f"{task_name}:{source_index}"
                prompt = task.doc_to_text(doc)
                visuals = task.doc_to_visual(doc)
                if not isinstance(visuals, list) or len(visuals) != 1 or not isinstance(visuals[0], Image.Image):
                    raise ValueError("E2 requires one PIL image per sample")
                answer, record = _generate(
                    config, model_name, spec, condition, lm, session,
                    sample_id, visuals[0], str(prompt), task.get_config("generation_kwargs") or {},
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
                rows.append(record)
                for name, value in processed.items():
                    if isinstance(value, (int, float, bool)):
                        metric_values.setdefault(name, []).append(value)
                with sample_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            metrics = {}
            aggregators = task.aggregation()
            for name, values in metric_values.items():
                aggregator = aggregators.get(name)
                metrics[name] = float(aggregator(values) if aggregator is not None else sum(values) / len(values))
            primary = _primary_metric(task_name, metrics)
            task_results[task_name] = {"metrics": metrics, "primary_score": primary, "samples": len(rows)}
        payload = {
            "model": spec.pretrained,
            "model_alias": model_name,
            "condition": condition.name,
            "tasks": task_results,
            "macro_average": sum(row["primary_score"] for row in task_results.values()) / len(task_results),
        }
        result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload
    finally:
        patch.remove()


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
    messages = _messages(lm, prepared_image, prompt)
    text = _chat_template(lm, messages, model_name)
    inputs = lm.processor(
        text=[text], images=[prepared_image], do_resize=False, return_tensors="pt"
    ).to(lm.device)
    actual_grid = tuple(int(value) for value in inputs["image_grid_thw"][0].tolist())
    expected_grid = (1, plan.grid_height * int(lm.model.config.vision_config.spatial_merge_size),
                     plan.grid_width * int(lm.model.config.vision_config.spatial_merge_size))
    if actual_grid != expected_grid:
        raise ValueError(f"processor grid {actual_grid} does not match E2 plan {expected_grid}")
    session.begin_sample(sample_id)
    kwargs = {
        "max_new_tokens": min(int(generation_kwargs.get("max_new_tokens", config.max_new_tokens)), config.max_new_tokens),
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
        output = lm.model.generate(**inputs, **kwargs)
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
    if diagnostics["active_tokens"] is None:
        diagnostics["active_tokens"] = plan.visual_tokens
        diagnostics["compression_ratio"] = plan.visual_tokens / high.visual_tokens
    return answer, diagnostics


def _task_manager(model_name: str):
    from lmms_eval.tasks import TaskManager
    return TaskManager(verbosity="ERROR", model_name="qwen_vl" if model_name == "qwen25" else "qwen3_5")


def _load_task(manager: Any, task_name: str) -> Any:
    from lmms_eval.tasks import get_task_dict
    task = get_task_dict([task_name], manager, "simple")[task_name]
    return task[1] if isinstance(task, tuple) else task


def load_tasks(model_name: str, task_names: tuple[str, ...]) -> dict[str, Any]:
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


def _chat_template(lm: Any, messages: list[dict[str, Any]], model_name: str) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if model_name == "qwen35":
        kwargs["enable_thinking"] = False
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
