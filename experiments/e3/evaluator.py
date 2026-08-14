from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm  # type: ignore[import-untyped]

from experiments.e2.config import ModelSpec
from experiments.e2.data import source_indices
from experiments.e2.image import ImagePlan, aligned_high_resolution, lowres_plan, resize
from experiments.e2.evaluator import load_tasks

from .conditions import Condition
from .config import E3Config
from .patch import install_e3
from .session import E3Session


ANALYSIS_INSTRUCTION = """Look carefully at the image and answer the question using the relevant visual evidence.
First, write a focused and natural analysis explaining how the image supports your conclusion. Keep it relevant, avoid unnecessary repetition, and do not exceed 200 words. Then give the final answer as briefly as possible.

Use exactly these labels:
Analyze: <your analysis>
Answer: <your final answer>"""
PROMPT_VERSION = "analyze_answer_v1"


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
    if sample_path.exists():
        sample_path.unlink()
    session = E3Session(condition, anchor_window=config.anchor_window)
    patch = install_e3(lm.model, session)
    try:
        task_results: dict[str, Any] = {}
        for task_name in config.tasks:
            task = tasks[task_name]
            documents = task.eval_docs
            metric_values: dict[str, list[Any]] = {}
            row_count = 0
            for source_index in tqdm(
                source_indices(config.e2_config(), task_name),
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
                for name, value in processed.items():
                    if isinstance(value, (int, float, bool)):
                        metric_values.setdefault(name, []).append(value)
                with sample_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                row_count += 1
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
                "samples": row_count,
            }
        payload = {
            "model": spec.pretrained,
            "model_alias": model_name,
            "condition": condition.name,
            "prompt_version": PROMPT_VERSION,
            "tasks": task_results,
            "macro_average": sum(row["primary_score"] for row in task_results.values())
            / len(task_results),
        }
        result_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return payload
    finally:
        patch.remove()


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
    return f"{prompt.rstrip()}\n\n{ANALYSIS_INSTRUCTION}"


def parse_response(response: str) -> ParsedResponse:
    raw = response.strip()
    analyze_match = re.search(
        r"(?ims)^\s*Analyze:\s*(.*?)\s*^\s*Answer:",
        raw,
    )
    answer_matches = re.findall(r"(?im)^\s*Answer:\s*(.*?)\s*$", raw)
    analyze = analyze_match.group(1).strip() if analyze_match else ""
    if answer_matches:
        answer = answer_matches[-1].strip()
    else:
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        answer = lines[-1] if lines else raw
        if answer.lower().startswith("answer:"):
            answer = answer.split(":", 1)[1].strip()
    word_count = len(re.findall(r"\b[\w'-]+\b", analyze, flags=re.UNICODE))
    return ParsedResponse(
        raw=raw,
        analyze=analyze,
        answer=answer,
        format_compliant=analyze_match is not None and bool(answer_matches),
        analyze_word_count=word_count,
    )


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
    return inputs


def _messages(lm: Any, image: Image.Image, prompt: str) -> list[dict[str, Any]]:
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
