from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from .conditions import Condition, Suite, conditions_for_suite
from .config import E4Config


def analyze(
    config: E4Config,
    model_name: str,
    *,
    suites: list[Suite] | None = None,
) -> list[dict[str, Any]]:
    selected_suites: list[Suite] = suites or ["formal", "reasoning", "compression"]
    root = config.output_dir / model_name
    all_results: dict[tuple[str, str], dict[str, Any]] = {}
    all_samples: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for suite in selected_suites:
        for condition in conditions_for_suite(suite):
            result_path = root / suite / condition.name / "results.json"
            sample_path = root / suite / condition.name / "samples.jsonl"
            if not result_path.is_file() or not sample_path.is_file():
                raise FileNotFoundError(f"E4 {suite}/{condition.name} is incomplete")
            all_results[(suite, condition.name)] = json.loads(
                result_path.read_text(encoding="utf-8")
            )
            all_samples[(suite, condition.name)] = _sample_map(sample_path)
    if "compression" in selected_suites and ("reasoning", "full") not in all_results:
        result_path = root / "reasoning" / "full" / "results.json"
        sample_path = root / "reasoning" / "full" / "samples.jsonl"
        if not result_path.is_file() or not sample_path.is_file():
            raise FileNotFoundError(
                "E4 compression analysis requires reasoning/full for the high-resolution prefill check"
            )
        all_results[("reasoning", "full")] = json.loads(result_path.read_text(encoding="utf-8"))
        all_samples[("reasoning", "full")] = _sample_map(sample_path)

    rows: list[dict[str, Any]] = []
    for suite in selected_suites:
        conditions = conditions_for_suite(suite)
        for condition in conditions:
            result = all_results[(suite, condition.name)]
            samples = all_samples[(suite, condition.name)]
            for task_name, task_result in result["tasks"].items():
                score = float(task_result["primary_score"])
                if not math.isfinite(score):
                    raise ValueError(f"non-finite E4 score for {suite}/{condition.name}/{task_name}")
                baseline_suite, baseline_name = _baseline(suite, condition)
                baseline_result = all_results.get((baseline_suite, baseline_name))
                baseline_samples = all_samples.get((baseline_suite, baseline_name))
                baseline_score = (
                    float(baseline_result["tasks"][task_name]["primary_score"])
                    if baseline_result is not None and task_name in baseline_result["tasks"]
                    else None
                )
                pair: dict[str, float | None] = (
                    _pair_metrics(baseline_samples, samples, task_name)
                    if baseline_samples is not None
                    else _empty_pair()
                )
                if condition.native:
                    prefill_suite = "reasoning" if suite == "compression" else suite
                    prefill_samples = all_samples.get((prefill_suite, "full"))
                    if prefill_samples is None:
                        raise FileNotFoundError(
                            f"E4 {suite} analysis requires {prefill_suite}/full"
                        )
                    prefill_pair = _pair_metrics(prefill_samples, samples, task_name)
                    pair["first_token_agreement"] = prefill_pair["first_token_agreement"]
                if condition.native and pair["first_token_agreement"] != 1.0:
                    raise ValueError(
                        f"E4 prefill mismatch for {suite}/{condition.name}/{task_name}: "
                        f"{pair['first_token_agreement']!r}"
                    )
                rows.append(
                    {
                        "suite": suite,
                        "condition": condition.name,
                        "task": task_name,
                        "score": score,
                        "baseline": f"{baseline_suite}/{baseline_name}",
                        "delta_baseline": score - baseline_score if baseline_score is not None else None,
                        **pair,
                        **_diagnostics(samples, task_name),
                    }
                )
            rows.extend(_visualprobe_group_rows(suite, condition, result, samples, all_results, all_samples))
            rows.append(
                {
                    "suite": suite,
                    "condition": condition.name,
                    "task": "macro_average",
                    "score": float(result["macro_average"]),
                    "baseline": None,
                    "delta_baseline": None,
                    **_empty_pair(),
                    **_diagnostics(samples, None),
                }
            )
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _baseline(suite: str, condition: Condition) -> tuple[str, str]:
    if suite == "compression":
        return ("compression", "lowres2") if condition.name != "lowres2" else ("reasoning", "full")
    if condition.name == "full":
        return suite, "full"
    if condition.name == "lowres4":
        return suite, "full"
    return suite, "lowres4"


def _pair_metrics(
    baseline: dict[str, dict[str, Any]],
    current: dict[str, dict[str, Any]],
    task: str | set[str],
) -> dict[str, float | None]:
    names = {task} if isinstance(task, str) else task
    keys = [key for key, row in current.items() if row["task"] in names and key in baseline]
    correct = [key for key in keys if baseline[key].get("correct") is True]
    first = []
    divergence = []
    matches = total = 0
    for key in keys:
        left = baseline[key]["generated_token_ids"]
        right = current[key]["generated_token_ids"]
        if left and right:
            first.append(left[0] == right[0])
        shared = min(len(left), len(right))
        matches += sum(left[index] == right[index] for index in range(shared))
        total += max(len(left), len(right))
        first_difference = next(
            (index for index in range(shared) if left[index] != right[index]),
            shared if len(left) != len(right) else -1,
        )
        if first_difference >= 0:
            divergence.append(first_difference)
    return {
        "control_correct_retention": (
            sum(current[key].get("correct") is True for key in correct) / len(correct)
            if correct
            else None
        ),
        "token_agreement": matches / total if total else None,
        "first_token_agreement": sum(first) / len(first) if first else None,
        "mean_first_divergence": sum(divergence) / len(divergence) if divergence else None,
    }


def _empty_pair() -> dict[str, float | None]:
    return {
        "control_correct_retention": None,
        "token_agreement": None,
        "first_token_agreement": None,
        "mean_first_divergence": None,
    }


def _diagnostics(
    samples: dict[str, dict[str, Any]], task: str | set[str] | None
) -> dict[str, Any]:
    names = {task} if isinstance(task, str) else task
    rows = [
        row for row in samples.values() if names is None or row["task"] in names
    ]
    result: dict[str, Any] = {}
    for name in (
        "highres_visual_tokens",
        "visual_tokens",
        "active_tokens",
        "compression_ratio",
        "retained_main_visual_tokens",
        "native_bank_tokens",
        "prefill_seconds",
        "decode_seconds",
        "native_prefill_seconds",
        "generation_seconds",
        "total_seconds",
        "analysis_word_count",
        "route_event_count",
        "route_swap_count",
        "mean_front_jaccard",
    ):
        values = [float(row[name]) for row in rows if row.get(name) is not None]
        result[name] = sum(values) / len(values) if values else None
    scales = {1: 0, 4: 0, 16: 0, 64: 0}
    for row in rows:
        for node in row.get("final_front", []):
            scale = int(node["area_scale"])
            if scale in scales:
                scales[scale] += 1
    total_nodes = sum(scales.values())
    result.update(
        {f"scale_{scale}_fraction": count / total_nodes if total_nodes else None for scale, count in scales.items()}
    )
    roi_values = [value for row in rows if (value := _roi_fine_gain(row)) is not None]
    result["roi_fine_gain"] = sum(roi_values) / len(roi_values) if roi_values else None
    return result


def _roi_fine_gain(row: dict[str, Any]) -> float | None:
    roi = row.get("roi")
    front = row.get("final_front")
    if not isinstance(roi, list) or len(roi) < 4 or not front:
        return None
    try:
        x0, y0, x1, y1 = map(float, roi[:4])
    except (TypeError, ValueError):
        return None
    width, height = map(float, row["original_size"])
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) > 1.0:
        x0, x1, y0, y1 = x0 / width, x1 / width, y0 / height, y1 / height
    grid_h, grid_w = row["highres_grid"]
    inside: list[float] = []
    outside: list[float] = []
    for node in front:
        cx = (node["x0"] + node["x1"]) / (2 * grid_w)
        cy = (node["y0"] + node["y1"]) / (2 * grid_h)
        density = 1.0 / float(node["area_scale"])
        (inside if x0 <= cx <= x1 and y0 <= cy <= y1 else outside).append(density)
    if not inside or not outside:
        return None
    return sum(inside) / len(inside) - sum(outside) / len(outside)


def _visualprobe_group_rows(
    suite: str,
    condition: Condition,
    result: dict[str, Any],
    samples: dict[str, dict[str, Any]],
    all_results: dict[tuple[str, str], dict[str, Any]],
    all_samples: dict[tuple[str, str], dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    names = [name for name in ("visualprobe_easy", "visualprobe_medium", "visualprobe_hard") if name in result["tasks"]]
    if not names:
        return []
    total = sum(result["tasks"][name]["samples"] for name in names)
    score = sum(result["tasks"][name]["primary_score"] * result["tasks"][name]["samples"] for name in names) / total
    baseline_suite, baseline_name = _baseline(suite, condition)
    baseline_result = all_results.get((baseline_suite, baseline_name))
    baseline_score = None
    baseline_samples = all_samples.get((baseline_suite, baseline_name))
    if baseline_result is not None and all(name in baseline_result["tasks"] for name in names):
        baseline_score = sum(
            baseline_result["tasks"][name]["primary_score"] * baseline_result["tasks"][name]["samples"]
            for name in names
        ) / sum(baseline_result["tasks"][name]["samples"] for name in names)
    pair = (
        _pair_metrics(baseline_samples, samples, set(names))
        if baseline_samples is not None
        else _empty_pair()
    )
    if condition.native:
        prefill_suite = "reasoning" if suite == "compression" else suite
        prefill_samples = all_samples.get((prefill_suite, "full"))
        if prefill_samples is not None:
            pair["first_token_agreement"] = _pair_metrics(
                prefill_samples, samples, set(names)
            )["first_token_agreement"]
    return [
        {
            "suite": suite,
            "condition": condition.name,
            "task": "visualprobe",
            "score": score,
            "baseline": f"{baseline_suite}/{baseline_name}",
            "delta_baseline": score - baseline_score if baseline_score is not None else None,
            **pair,
            **_diagnostics(samples, set(names)),
        }
    ]


def _sample_map(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        sample_id = str(row["sample_id"])
        if sample_id in rows:
            raise ValueError(f"duplicate E4 sample ID {sample_id!r}")
        rows[sample_id] = row
    return rows
