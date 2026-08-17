from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm  # type: ignore[import-untyped]

load_dotenv()
if os.getenv("OPENAI_API_KEY"):
    os.environ.setdefault("JUDGE_API_KEY", os.environ["OPENAI_API_KEY"])
    os.environ.setdefault("JUDGE_BASE_URL", os.getenv("OPENAI_API_URL", ""))
    os.environ.setdefault("JUDGE_MODEL_NAME", os.getenv("MODEL_VERSION", "gpt-4o-mini"))
    os.environ.setdefault("USE_LLM_JUDGE", "True")

from experiments.e2.config import ModelSpec
from experiments.e2.data import source_indices
from experiments.e2.image import ImagePlan, aligned_high_resolution, lowres_plan, resize
from experiments.e2.evaluator import _validate_image_token_count, load_tasks
from lmms_eval.tasks._task_utils.reasoning_utils import (
    JUDGE_MAX_TOKENS,
    JUDGE_THINKING,
    MODEL_NAME,
    USE_LLM_JUDGE,
    compute_score,
    extract_anwser_tag,
    format_reward,
    llm_as_judge_sync,
)

from .conditions import Condition
from .config import E3Config
from .patch import install_e3
from .session import E3Session


PROMPT_VERSION = "lmms_reasoning_v6"
SCORING_VERSION = "llm_judge_invalid_format_v1"
ANSWER_LENGTH_INSTRUCTION = "Answer the question using a single word or phrase."
ANSWER_WORD_INSTRUCTION = "Answer the question with a single word."
REASONING_SYSTEM_PROMPT = (
    "You are a helpful assistant. When the user asks a question, your response must include two "
    "parts: first, the reasoning process enclosed in `<analysis>...</analysis>` tags, followed by "
    "a clear, concise final answer enclosed in `<answer>...</answer>` tags that directly addresses "
    "the question and contains only the short final answer without explanation."
)


@dataclass(frozen=True, slots=True)
class ParsedResponse:
    raw: str
    analyze: str
    answer: str
    format_compliant: bool
    analyze_word_count: int

    @property
    def analyze_within_limit(self) -> bool:
        return self.analyze_word_count <= 200


def evaluate_condition(
    config: E3Config,
    model_name: str,
    spec: ModelSpec,
    condition: Condition,
    lm: Any,
    output_dir: Path,
    tasks: dict[str, Any],
    *,
    scoring_version: str = SCORING_VERSION,
) -> dict[str, Any]:
    condition_dir = output_dir / condition.name
    condition_dir.mkdir(parents=True, exist_ok=True)
    result_path = condition_dir / "results.json"
    sample_path = condition_dir / "samples.jsonl"
    if result_path.exists() and sample_path.exists():
        cached = json.loads(result_path.read_text(encoding="utf-8"))
        if cached.get("prompt_version") != PROMPT_VERSION:
            raise ValueError(
                f"existing E3 output for {condition.name} uses an incompatible prompt version"
            )
        return cached
    task_indices = {
        task_name: source_indices(config.e2_config(), task_name)
        for task_name in config.tasks
    }
    expected_sample_ids = {
        f"{task_name}:{source_index}"
        for task_name, indices in task_indices.items()
        for source_index in indices
    }
    completed_samples = _read_completed_samples(sample_path, expected_sample_ids)
    session = E3Session(condition, anchor_window=config.anchor_window)
    patch = install_e3(lm.model, session)
    try:
        task_results: dict[str, Any] = {}
        for task_name in config.tasks:
            task = tasks[task_name]
            documents = task.eval_docs
            indices = task_indices[task_name]
            sample_ids = [f"{task_name}:{source_index}" for source_index in indices]
            rows = [completed_samples[sample_id] for sample_id in sample_ids if sample_id in completed_samples]
            metric_values: dict[str, list[Any]] = {}
            for record in rows:
                for name, value in record["metrics"].items():
                    if isinstance(value, (int, float, bool)):
                        metric_values.setdefault(name, []).append(value)
            pending_indices = [
                source_index
                for source_index, sample_id in zip(indices, sample_ids)
                if sample_id not in completed_samples
            ]
            for source_index in tqdm(
                pending_indices,
                total=len(indices),
                initial=len(rows),
                desc=f"{model_name} {condition.name} {task_name}",
            ):
                doc = documents[source_index]
                sample_id = f"{task_name}:{source_index}"
                prompt = task.doc_to_text(doc)
                visuals = task.doc_to_visual(doc)
                if (
                    not isinstance(visuals, list)
                    or len(visuals) != 1
                    or not isinstance(visuals[0], Image.Image)
                ):
                    raise ValueError("E3 requires one PIL image per sample")
                answer, record = _generate(
                    config,
                    model_name,
                    spec,
                    condition,
                    lm,
                    session,
                    sample_id,
                    visuals[0],
                    str(prompt),
                    task.get_config("generation_kwargs") or {},
                )
                processed = _score_response(task_name, doc, str(prompt), record["raw_prediction"])
                record.update(
                    {
                        "task": task_name,
                        "source_index": source_index,
                        "prediction": answer,
                        "target": task.doc_to_target(doc),
                        "metrics": _jsonable(processed),
                        "scoring_version": scoring_version,
                    }
                )
                rows.append(record)
                completed_samples[sample_id] = record
                for name, value in processed.items():
                    if isinstance(value, (int, float, bool)):
                        metric_values.setdefault(name, []).append(value)
                with sample_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            metrics = {}
            aggregators = task.aggregation()
            for name, values in metric_values.items():
                aggregator = aggregators.get(name)
                metrics[name] = float(
                    aggregator(values) if aggregator is not None else sum(values) / len(values)
                )
            task_results[task_name] = {
                "metrics": metrics,
                "primary_score": _primary_metric(task_name, metrics),
                "samples": len(rows),
            }
        payload = _result_payload(
            spec.pretrained, model_name, condition, task_results, scoring_version
        )
        result_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return payload
    finally:
        patch.remove()


def _read_completed_samples(path: Path, expected_ids: set[str]) -> dict[str, dict[str, Any]]:
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
            raise ValueError(f"invalid E3 sample ID in {path}: {sample_id!r}")
        if sample_id in completed:
            raise ValueError(f"duplicate E3 sample ID in {path}: {sample_id!r}")
        if not isinstance(row.get("metrics"), dict):
            raise ValueError(f"missing E3 metrics in {path}: {sample_id!r}")
        completed[sample_id] = row
    return completed


def reevaluate_condition(
    config: E3Config,
    model_name: str,
    spec: ModelSpec,
    condition: Condition,
    output_dir: Path,
    tasks: dict[str, Any],
    *,
    scoring_version: str = SCORING_VERSION,
    restart: bool = False,
    workers: int = 8,
) -> dict[str, Any]:
    condition_dir = output_dir / condition.name
    sample_path = condition_dir / "samples.jsonl"
    result_path = condition_dir / "results.json"
    task_indices = {
        task_name: source_indices(config.e2_config(), task_name)
        for task_name in config.tasks
    }
    expected_ids = {
        f"{task_name}:{source_index}"
        for task_name, indices in task_indices.items()
        for source_index in indices
    }
    samples = _read_completed_samples(sample_path, expected_ids)
    if len(samples) != len(expected_ids):
        raise ValueError(
            f"cannot reevaluate incomplete E3 condition {condition.name!r}: "
            f"found {len(samples)} of {len(expected_ids)} samples"
        )

    task_results: dict[str, Any] = {}
    for task_name in config.tasks:
        task = tasks[task_name]
        metric_values: dict[str, list[float]] = {}
        completed = sum(
            samples[f"{task_name}:{source_index}"].get("scoring_version") == scoring_version
            for source_index in task_indices[task_name]
        )
        pending_indices = [
            source_index
            for source_index in task_indices[task_name]
            if restart
            or samples[f"{task_name}:{source_index}"].get("scoring_version") != scoring_version
        ]
        requests: dict[int, tuple[Any, str, str]] = {}
        for source_index in pending_indices:
            sample_id = f"{task_name}:{source_index}"
            raw_prediction = samples[sample_id].get("raw_prediction")
            if not isinstance(raw_prediction, str):
                raise ValueError(f"missing raw_prediction for {sample_id}")
            doc = task.eval_docs[source_index]
            requests[source_index] = (doc, str(task.doc_to_text(doc)), raw_prediction)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _score_response,
                    task_name,
                    doc,
                    prompt,
                    raw_prediction,
                ): source_index
                for source_index, (doc, prompt, raw_prediction) in requests.items()
            }
            for future in tqdm(
                as_completed(futures),
                total=len(task_indices[task_name]),
                initial=0 if restart else completed,
                desc=f"reevaluate {model_name} {condition.name} {task_name}",
            ):
                source_index = futures[future]
                sample_id = f"{task_name}:{source_index}"
                samples[sample_id]["metrics"] = _jsonable(future.result())
                samples[sample_id]["scoring_version"] = scoring_version
                _write_jsonl(sample_path, list(samples.values()))
        for source_index in task_indices[task_name]:
            metrics = samples[f"{task_name}:{source_index}"]["metrics"]
            for name, value in metrics.items():
                metric_values.setdefault(name, []).append(float(value))
        aggregators = task.aggregation()
        metrics = {}
        for name, values in metric_values.items():
            aggregator = aggregators.get(name)
            metrics[name] = float(aggregator(values) if aggregator is not None else sum(values) / len(values))
        task_results[task_name] = {
            "metrics": metrics,
            "primary_score": _primary_metric(task_name, metrics),
            "samples": len(task_indices[task_name]),
        }

    _write_jsonl(sample_path, list(samples.values()))
    payload = _result_payload(
        spec.pretrained, model_name, condition, task_results, scoring_version
    )
    _write_json(result_path, payload)
    return payload


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _result_payload(
    pretrained: str,
    model_name: str,
    condition: Condition,
    task_results: dict[str, Any],
    scoring_version: str = SCORING_VERSION,
) -> dict[str, Any]:
    return {
        "model": pretrained,
        "model_alias": model_name,
        "condition": condition.name,
        "prompt_version": PROMPT_VERSION,
        "scoring_version": scoring_version,
        "tasks": task_results,
        "macro_average": sum(row["primary_score"] for row in task_results.values())
        / len(task_results),
    }


def _generate(
    config: E3Config,
    model_name: str,
    spec: ModelSpec,
    condition: Condition,
    lm: Any,
    session: E3Session,
    sample_id: str,
    image: Image.Image,
    prompt: str,
    generation_kwargs: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    high = aligned_high_resolution(image, spec.pixel_per_token, spec.min_pixels, spec.max_pixels)
    plan = lowres_plan(high, 2) if condition.representation == "lowres2" else high
    structured_prompt = build_analysis_prompt(prompt)
    session.begin_sample(sample_id)
    if condition.native:
        try:
            auxiliary_plan = lowres_plan(high, 2)
            auxiliary_inputs = _prepare_inputs(
                lm,
                model_name,
                resize(image, auxiliary_plan),
                structured_prompt,
                auxiliary_plan,
            )
            session.begin_native_capture(4)
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
    inputs = _prepare_inputs(lm, model_name, resize(image, plan), structured_prompt, plan)
    kwargs = {
        # E3 owns the response format and must not inherit short-answer task caps.
        "max_new_tokens": config.max_new_tokens,
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
    raw_answer = lm.processor.decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    parsed = parse_response(raw_answer)
    diagnostics = session.diagnostics()
    diagnostics.update(
        {
            "representation": condition.representation,
            "anchor_window": config.anchor_window,
            "raw_prediction": parsed.raw,
            "analyze": parsed.analyze,
            "format_compliant": parsed.format_compliant,
            "analyze_word_count": parsed.analyze_word_count,
            "analyze_within_limit": parsed.analyze_within_limit,
            "input_size": [plan.height, plan.width],
            "grid": [plan.grid_height, plan.grid_width],
            "visual_tokens": plan.visual_tokens,
            "highres_visual_tokens": high.visual_tokens,
            "active_tokens": session.front.node_count if session.front is not None else plan.visual_tokens,
            "compression_ratio": (
                session.front.node_count / high.visual_tokens
                if session.front is not None
                else plan.visual_tokens / high.visual_tokens
            ),
            "generated_token_ids": generated.detach().cpu().tolist(),
            "score_top1_token_ids": [int(scores[0].argmax()) for scores in output.scores],
            "total_seconds": elapsed,
        }
    )
    return parsed.answer, diagnostics


def build_analysis_prompt(prompt: str) -> str:
    return prompt.replace(ANSWER_LENGTH_INSTRUCTION, "").replace(ANSWER_WORD_INSTRUCTION, "").strip()


def parse_response(response: str) -> ParsedResponse:
    raw = response.strip()
    format_compliant = format_reward(raw) == 1.0
    answer = extract_anwser_tag(raw).strip() if format_compliant else ""
    analysis_match = re.search(r"<(?:analysis|think)>(.*?)</(?:analysis|think)>", raw, flags=re.DOTALL)
    analyze = analysis_match.group(1).strip() if analysis_match else ""
    word_count = len(re.findall(r"\b[\w'-]+\b", analyze, flags=re.UNICODE))
    return ParsedResponse(
        raw=raw,
        analyze=analyze,
        answer=answer,
        format_compliant=format_compliant,
        analyze_word_count=word_count,
    )


def _score_response(task_name: str, doc: Any, question: str, response: str) -> dict[str, float]:
    parsed = parse_response(response)
    metric = "relaxed_overall" if task_name == "chartqa_lite" else "exact_match"
    if not parsed.format_compliant:
        score = (
            llm_as_judge_sync(response, _ground_truth(task_name, doc), {"question": question})
            if USE_LLM_JUDGE == "True"
            else 0.0
        )
        return {metric: float(score), "format_score": 0.0}
    score = compute_score(
        data_source=task_name,
        solution_str=response,
        ground_truth=_ground_truth(task_name, doc),
        extra_info={"question": question},
    )
    return {
        metric: float(score["acc_score"]),
        "format_score": float(score["format_reward_score"]),
    }


def llm_judge_status() -> tuple[bool, str, int, str]:
    return USE_LLM_JUDGE == "True", MODEL_NAME, JUDGE_MAX_TOKENS, JUDGE_THINKING


def _ground_truth(task_name: str, doc: Any) -> str:
    if task_name == "vqav2_val_lite":
        return str(doc["multiple_choice_answer"])
    if task_name == "textvqa_val_lite":
        answers = [str(answer) for answer in doc["answers"]]
        return Counter(answers).most_common(1)[0][0]
    return str(doc["answer"])


def _prepare_inputs(
    lm: Any,
    model_name: str,
    image: Image.Image,
    prompt: str,
    plan: ImagePlan,
) -> Any:
    messages = _messages(lm, image, prompt)
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if model_name == "qwen35":
        kwargs["enable_thinking"] = False
    text = lm.processor.apply_chat_template(messages, **kwargs)
    inputs = lm.processor(
        text=[text], images=[image], do_resize=False, return_tensors="pt"
    ).to(lm.device)
    actual_grid = tuple(int(value) for value in inputs["image_grid_thw"][0].tolist())
    merge_size = int(lm.model.config.vision_config.spatial_merge_size)
    expected_grid = (1, plan.grid_height * merge_size, plan.grid_width * merge_size)
    if actual_grid != expected_grid:
        raise ValueError(f"processor grid {actual_grid} does not match E3 plan {expected_grid}")
    _validate_image_token_count(inputs, lm.model)
    return inputs


def _messages(lm: Any, image: Image.Image, prompt: str) -> list[dict[str, Any]]:
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": REASONING_SYSTEM_PROMPT}],
        }
    ]
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    )
    return messages


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
