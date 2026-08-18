from __future__ import annotations

import json
import re
import time
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import torch
from PIL import Image
from tqdm import tqdm  # type: ignore[import-untyped]

from experiments.distributed import distributed_context, merge_rank_jsonl, merge_rank_text
from experiments.local_datasets import use_local_lmms_datasets
from tokenfovea.config import FoveaConfig
from tokenfovea.integrations.qwen import install_tokenfovea
from tokenfovea.session import FoveaSession

from .conditions import Condition, Suite
from .config import E4Config
from .data import suite_indices
from .image import aligned_high_resolution, resize, scaled_plan, visual_tokens
from .runtime import ForwardTimer, RouteTraceObserver
from .session import E4Session

REASONING_PROMPT = (
    "Analyze the image and question carefully. Give only the evidence needed in at most "
    "200 words, then put the final answer alone inside <answer>...</answer>."
)


def load_tasks(model_name: str, task_names: tuple[str, ...]) -> dict[str, Any]:
    from lmms_eval.tasks import TaskManager, get_task_dict

    lmms_name = "qwen2_5_vl" if model_name == "qwen25" else "qwen3_5"
    task_root = str(Path(__file__).parent / "tasks")
    with use_local_lmms_datasets():
        with _selected_splits_only():
            manager = TaskManager(
                verbosity="ERROR",
                model_name=lmms_name,
                include_path=task_root,
            )
            loaded = {}
            for name in task_names:
                task = get_task_dict([name], manager, "simple")[name]
                loaded[name] = task[1] if isinstance(task, tuple) else task
            return loaded


@contextmanager
def _selected_splits_only() -> Iterator[None]:
    """Prevent lmms-eval from downloading unused splits of large E4 datasets."""
    import datasets  # type: ignore[import-untyped]

    original = datasets.load_dataset
    selected = {
        ("DreamMr/HR-Bench", "hrbench_version_split"): "hrbench_8k",
        ("initiacms/XLRS-Bench-lite", None): "train",
        ("lmms-lab-encoder/vstar-bench", None): "test",
        ("Mini-o3/VisualProbe_Easy", None): "validation",
        ("Mini-o3/VisualProbe_Medium", None): "validation",
        ("Mini-o3/VisualProbe_Hard", None): "validation",
        ("Jiazuo98/Finers-4k-benchmark", None): "test",
        ("Wenliang04/HRScene", "realworld_combined"): "testmini",
        ("Lin-Chen/MMStar", None): "val",
        ("lmms-lab-encoder/ChartQA", None): "test",
        ("lmms-lab-encoder/textvqa", None): "validation",
    }

    def load_selected(path: str, name: str | None = None, *args: Any, **kwargs: Any):
        split = selected.get((path, name))
        if split is None or kwargs.get("split") is not None:
            return original(path, name, *args, **kwargs)
        dataset = original(path, name, *args, split=split, **kwargs)
        return datasets.DatasetDict({split: dataset})

    datasets.load_dataset = load_selected
    try:
        yield
    finally:
        datasets.load_dataset = original


def make_fovea_config(
    condition: Condition,
    budget: int,
    signal_selection: Path,
) -> FoveaConfig:
    if not condition.native:
        raise ValueError("only Native E4 conditions use FoveaConfig")
    mode: Literal["uniform", "dynamic"] = (
        "uniform" if condition.kind == "uniform" else "dynamic"
    )
    return FoveaConfig(
        budget=budget,
        mode=mode,
        pooling_mode="native_multiscale",
        position_mode="native_center",
        signal_selection=str(signal_selection) if condition.use_top8 else None,
        signal_aggregation="mean",
        update_interval=1_000_000 if condition.kind == "static" else 1,
        max_swaps=100,
        epsilon=0.05,
        attention_ema=0.0,
        score_mode="mass",
        route_after_prefill=True,
    )


def evaluate_condition(
    config: E4Config,
    model_name: str,
    condition: Condition,
    suite: Suite,
    lm: Any,
    tasks: dict[str, Any],
    output_dir: Path,
    *,
    task_name: str | None = None,
) -> dict[str, Any]:
    distributed = distributed_context()
    condition_dir = output_dir / suite / condition.name
    condition_dir.mkdir(parents=True, exist_ok=True)
    result_path = condition_dir / "results.json"
    sample_path = condition_dir / "samples.jsonl"
    trace_path = condition_dir / "route_traces.jsonl"
    if distributed.is_main:
        merge_rank_jsonl(sample_path, key="sample_id")
        merge_rank_text(trace_path)
    distributed.barrier()
    indices = suite_indices(config, suite)
    if task_name is not None and task_name not in indices:
        raise ValueError(f"task {task_name!r} is not part of E4 suite {suite!r}")
    run_indices = {task_name: indices[task_name]} if task_name is not None else indices
    expected_ids = {
        f"{task}:{source_index}"
        for task, task_indices in indices.items()
        for source_index in task_indices
    }
    completed = _read_samples(sample_path, expected_ids)
    if result_path.exists() and len(completed) == len(expected_ids):
        return json.loads(result_path.read_text(encoding="utf-8"))
    if distributed.is_main and result_path.exists():
        result_path.unlink()
    distributed.barrier()
    write_samples = distributed.rank_path(sample_path) if distributed.enabled else sample_path
    write_traces = distributed.rank_path(trace_path) if distributed.enabled else trace_path

    observer = RouteTraceObserver()
    session: FoveaSession | None = None
    patch = None
    if condition.native:
        fovea_config = make_fovea_config(
            condition,
            budget=1,
            signal_selection=config.head_selections[model_name],
        )
        session = E4Session(
            fovea_config,
            prefill_static=condition.kind == "static",
            route_observer=observer if condition.routed else None,
        )
        patch = install_tokenfovea(lm.model, session)
    timer = ForwardTimer(
        lm.model,
        native_scale=lambda: session.native_capture_scale if session is not None else None,
    ).install()
    try:
        for current_task, task_indices in run_indices.items():
            task = tasks[current_task]
            pending = [
                index
                for index in task_indices
                if f"{current_task}:{index}" not in completed
            ]
            local = distributed.shard(pending)
            for source_index in tqdm(
                local,
                total=len(local),
                desc=f"E4 {model_name} {suite}/{condition.name} {current_task}",
            ):
                doc = task.eval_docs[source_index]
                sample_id = f"{current_task}:{source_index}"
                prompt = str(task.doc_to_text(doc))
                if suite != "formal":
                    prompt = f"{prompt.rstrip()}\n\n{REASONING_PROMPT}"
                visuals = task.doc_to_visual(doc)
                if (
                    not isinstance(visuals, list)
                    or len(visuals) != 1
                    or not isinstance(visuals[0], Image.Image)
                ):
                    raise ValueError("E4 requires exactly one PIL image per sample")
                observer.begin_sample(sample_id)
                timer.reset()
                prediction, record, traces = _generate(
                    config,
                    model_name,
                    condition,
                    suite,
                    lm,
                    session,
                    timer,
                    observer,
                    sample_id,
                    visuals[0],
                    prompt,
                    task.get_config("generation_kwargs") or {},
                )
                metrics = task.process_results(doc, [prediction])
                record.update(
                    {
                        "task": current_task,
                        "source_index": source_index,
                        "prediction": prediction,
                        "target": _jsonable(task.doc_to_target(doc)),
                        "metrics": _jsonable(metrics),
                        "correct": _metric_correct(metrics),
                        "roi": _extract_roi(doc),
                    }
                )
                completed[sample_id] = record
                with write_samples.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                if traces:
                    with write_traces.open("a", encoding="utf-8") as stream:
                        for row in traces:
                            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        distributed.barrier()
        if distributed.is_main:
            merge_rank_jsonl(sample_path, key="sample_id")
            merge_rank_text(trace_path)
            completed = _read_samples(sample_path, expected_ids)
            if len(completed) == len(expected_ids):
                payload = _aggregate(
                    config, model_name, condition, suite, tasks, indices, completed
                )
                result_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )
        distributed.barrier()
        if result_path.exists():
            return json.loads(result_path.read_text(encoding="utf-8"))
        return {
            "model_alias": model_name,
            "suite": suite,
            "condition": condition.name,
            "status": "partial",
        }
    finally:
        timer.remove()
        if patch is not None:
            patch.remove()


def _generate(
    config: E4Config,
    model_name: str,
    condition: Condition,
    suite: Suite,
    lm: Any,
    session: FoveaSession | None,
    timer: ForwardTimer,
    observer: RouteTraceObserver,
    sample_id: str,
    image: Image.Image,
    prompt: str,
    generation_kwargs: dict[str, Any],
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    spec = config.models[model_name]
    high = aligned_high_resolution(
        image, spec.pixel_per_token, spec.min_pixels, config.visual_token_cap
    )
    high_tokens = visual_tokens(high)
    target_tokens = high_tokens // condition.budget_area
    input_plan = scaled_plan(high, condition.lowres_divisor or 1)
    if condition.kind == "lowres" and visual_tokens(input_plan) != target_tokens:
        raise ValueError(
            f"E4 LowRes/Native target mismatch for {sample_id}: "
            f"{visual_tokens(input_plan)} != {target_tokens}"
        )
    if condition.kind == "full":
        target_tokens = high_tokens

    if session is not None:
        session.config.budget = target_tokens
        session.begin_native_sample()
        try:
            for divisor, area_scale in ((2, 4), (4, 16), (8, 64)):
                auxiliary_plan = scaled_plan(high, divisor)
                auxiliary = _prepare_inputs(
                    lm, model_name, resize(image, auxiliary_plan), prompt, auxiliary_plan
                )
                session.begin_native_capture(area_scale)
                with torch.inference_mode():
                    output = lm.model(**auxiliary, use_cache=True, return_dict=True)
                del output
                session.end_native_capture()
        except Exception:
            session.abort_native_sample()
            raise

    prepared = resize(image, input_plan)
    inputs = _prepare_inputs(lm, model_name, prepared, prompt, input_plan)
    task_limit = int(generation_kwargs.get("max_new_tokens", config.formal_max_new_tokens))
    max_new_tokens = (
        config.reasoning_max_new_tokens
        if suite != "formal"
        else min(task_limit, config.formal_max_new_tokens)
    )
    kwargs = {
        "max_new_tokens": max_new_tokens,
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
    generation_seconds = time.perf_counter() - started
    prompt_length = inputs["input_ids"].shape[-1]
    generated = output.sequences[0, prompt_length:]
    raw = lm.processor.decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    ).strip()
    until = generation_kwargs.get("until", [])
    if isinstance(until, str):
        until = [until]
    for term in until:
        if term:
            raw = raw.split(term)[0].strip()
    prediction, analysis, compliant = (
        parse_reasoning_response(raw) if suite != "formal" else (raw, "", True)
    )
    score_tokens = [int(scores[0].argmax()) for scores in output.scores]
    traces = observer.drain()
    final_front = _final_front(session)
    active_tokens = len(final_front) if final_front else visual_tokens(input_plan)
    if active_tokens != target_tokens:
        raise RuntimeError(
            f"E4 condition {condition.name} changed its fixed budget: {active_tokens} != {target_tokens}"
        )
    native_bank_tokens = (
        high_tokens // 4 + high_tokens // 16 + high_tokens // 64
        if condition.native
        else 0
    )
    record = {
        "sample_id": sample_id,
        "raw_prediction": raw,
        "analysis": analysis,
        "analysis_word_count": len(analysis.split()),
        "format_compliant": compliant,
        "original_size": list(image.size),
        "input_size": [input_plan.width, input_plan.height],
        "highres_grid": [high.grid_height, high.grid_width],
        "input_grid": [input_plan.grid_height, input_plan.grid_width],
        "highres_visual_tokens": high_tokens,
        "visual_tokens": visual_tokens(input_plan),
        "target_visual_tokens": target_tokens,
        "active_tokens": active_tokens,
        "compression_ratio": active_tokens / high_tokens,
        "retained_main_visual_tokens": high_tokens if condition.native else visual_tokens(input_plan),
        "native_bank_tokens": native_bank_tokens,
        "generated_token_ids": generated.detach().cpu().tolist(),
        "score_top1_token_ids": score_tokens,
        "generation_seconds": generation_seconds,
        "final_front": final_front,
        "route_event_count": len(traces),
        "route_swap_count": sum(int(row["swaps"]) for row in traces),
        "mean_front_jaccard": (
            sum(float(row["front_jaccard"]) for row in traces) / len(traces)
            if traces
            else None
        ),
        **timer.diagnostics(),
    }
    record["total_seconds"] = generation_seconds + float(record["native_prefill_seconds"])
    return prediction, record, traces


def _prepare_inputs(
    lm: Any,
    model_name: str,
    image: Image.Image,
    prompt: str,
    plan: Any,
) -> Any:
    messages = []
    if getattr(lm, "system_prompt", None):
        messages.append({"role": "system", "content": lm.system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    )
    template_kwargs = {"tokenize": False, "add_generation_prompt": True}
    if model_name == "qwen35":
        template_kwargs["enable_thinking"] = False
    text = lm.processor.apply_chat_template(messages, **template_kwargs)
    inputs = lm.processor(
        text=[text], images=[image], do_resize=False, return_tensors="pt"
    ).to(lm.device)
    merge = int(lm.model.config.vision_config.spatial_merge_size)
    actual = tuple(int(value) for value in inputs["image_grid_thw"][0].tolist())
    expected = (1, plan.grid_height * merge, plan.grid_width * merge)
    if actual != expected:
        raise ValueError(f"E4 processor grid {actual} does not match planned grid {expected}")
    image_tokens = int((inputs["input_ids"] == lm.model.config.image_token_id).sum())
    if image_tokens != visual_tokens(plan):
        raise ValueError(f"E4 input contains {image_tokens} visual tokens, expected {visual_tokens(plan)}")
    return inputs


def parse_reasoning_response(raw: str) -> tuple[str, str, bool]:
    answers = re.findall(r"<answer>\s*(.*?)\s*</answer>", raw, flags=re.DOTALL | re.IGNORECASE)
    analyses = re.findall(
        r"<(?:analysis|analyze)>\s*(.*?)\s*</(?:analysis|analyze)>",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if answers:
        before = raw[: raw.lower().rfind("<answer>")]
        analysis = analyses[-1].strip() if analyses else re.sub(r"<[^>]+>", "", before).strip()
        return answers[-1].strip(), analysis, True
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    return (lines[-1] if lines else ""), raw.strip(), False


def _final_front(session: FoveaSession | None) -> list[dict[str, int]]:
    if session is None or session.router is None or session.forest is None:
        return []
    ids = sorted(int(value) for value in session.router.active_ids().detach().cpu().tolist())
    return [
        {
            "node_id": node_id,
            "image_index": session.forest.node(node_id).image_index,
            "y0": session.forest.node(node_id).y0,
            "x0": session.forest.node(node_id).x0,
            "y1": session.forest.node(node_id).y1,
            "x1": session.forest.node(node_id).x1,
            "area_scale": session.forest.node(node_id).valid_count,
        }
        for node_id in ids
    ]


def _aggregate(
    config: E4Config,
    model_name: str,
    condition: Condition,
    suite: Suite,
    tasks: dict[str, Any],
    indices: dict[str, list[int]],
    samples: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    task_results: dict[str, dict[str, Any]] = {}
    for task_name, task_indices in indices.items():
        rows = [samples[f"{task_name}:{index}"] for index in task_indices]
        if len(rows) != len(task_indices):
            raise RuntimeError(f"E4 {condition.name}/{task_name} has incomplete results")
        values: dict[str, list[Any]] = {}
        for row in rows:
            for metric, value in row["metrics"].items():
                values.setdefault(metric, []).append(value)
        aggregators = tasks[task_name].aggregation()
        metrics = {}
        for metric, metric_values in values.items():
            aggregator = aggregators.get(metric)
            result = aggregator(metric_values) if aggregator is not None else sum(metric_values) / len(metric_values)
            metrics[metric] = float(result)
        primary = config.primary_metrics[task_name]
        if primary not in metrics:
            raise ValueError(f"E4 task {task_name} did not return primary metric {primary}")
        task_results[task_name] = {
            "metrics": metrics,
            "primary_score": metrics[primary],
            "samples": len(rows),
        }
    visual_names = [
        name
        for name in ("visualprobe_easy", "visualprobe_medium", "visualprobe_hard")
        if name in task_results
    ]
    scores = [
        float(value["primary_score"])
        for name, value in task_results.items()
        if name not in visual_names
    ]
    if visual_names:
        visual_count = sum(int(task_results[name]["samples"]) for name in visual_names)
        scores.append(
            sum(
                float(task_results[name]["primary_score"])
                * int(task_results[name]["samples"])
                for name in visual_names
            )
            / visual_count
        )
    return {
        "model": config.models[model_name].pretrained,
        "model_alias": model_name,
        "suite": suite,
        "condition": condition.name,
        "tasks": task_results,
        "macro_average": sum(scores) / len(scores),
    }


def _read_samples(path: Path, expected_ids: set[str]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = {}
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
        if sample_id not in expected_ids or sample_id in rows:
            raise ValueError(f"invalid or duplicate E4 sample ID: {sample_id!r}")
        rows[sample_id] = row
    return rows


def _metric_correct(metrics: dict[str, Any]) -> bool | None:
    numeric = list(_score_leaves(metrics))
    return max(numeric) > 0 if numeric else None


def _score_leaves(value: Any, key: str = ""):
    if isinstance(value, bool):
        yield float(value)
    elif isinstance(value, (int, float)) and any(
        marker in key.lower() for marker in ("score", "correct", "acc", "match")
    ):
        yield float(value)
    elif isinstance(value, dict):
        for name, item in value.items():
            yield from _score_leaves(item, str(name))
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _score_leaves(item, key)


def _extract_roi(doc: Any) -> Any:
    annotation = doc.get("annotations", doc) if hasattr(doc, "get") else {}
    for source in (annotation, doc):
        if not hasattr(source, "get"):
            continue
        for key in ("bbox", "target_bbox", "box"):
            value = source.get(key)
            if value not in (None, "", []):
                return _jsonable(value)
    return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
