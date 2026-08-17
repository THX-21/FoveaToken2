from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from .conditions import CONDITIONS
from .config import E2Config
from .runner import run_name


LOWRES_BASELINE = {
    "uniform2_kv_center": "lowres_2",
    "uniform2_hidden_center": "lowres_2",
    "uniform2_postrope": "lowres_2",
    "native_uniform4": "lowres_2",
    "uniform4_kv_center": "lowres_4",
    "uniform4_hidden_center": "lowres_4",
    "uniform4_postrope": "lowres_4",
    "native_uniform16": "lowres_4",
    "random_fixed_kv_center": "lowres_random_matched",
    "random_fixed_hidden_center": "lowres_random_matched",
    "random_fixed_postrope": "lowres_random_matched",
    "random_fixed_native": "lowres_random_matched",
    "random_perstep_kv_center": "lowres_random_matched",
    "random_perstep_hidden_center": "lowres_random_matched",
    "random_perstep_postrope": "lowres_random_matched",
    "random_perstep_native": "lowres_random_matched",
}

NATIVE_POOLING_BASELINES = {
    "native_uniform4": (
        "uniform2_kv_center", "uniform2_hidden_center", "uniform2_postrope"
    ),
    "native_uniform16": (
        "uniform4_kv_center", "uniform4_hidden_center", "uniform4_postrope"
    ),
    "random_fixed_native": (
        "random_fixed_kv_center", "random_fixed_hidden_center", "random_fixed_postrope"
    ),
    "random_perstep_native": (
        "random_perstep_kv_center", "random_perstep_hidden_center", "random_perstep_postrope"
    ),
}


def analyze(config: E2Config, model_name: str, *, thinking: bool = False) -> list[dict[str, Any]]:
    root = config.output_dir / run_name(model_name, thinking)
    results: dict[str, Any] = {}
    samples: dict[str, dict[str, dict[str, Any]]] = {}
    for condition in CONDITIONS:
        result_path = root / condition.name / "results.json"
        sample_path = root / condition.name / "samples.jsonl"
        if not result_path.exists() or not sample_path.exists():
            continue
        results[condition.name] = json.loads(result_path.read_text(encoding="utf-8"))
        samples[condition.name] = _sample_map(sample_path)
        expected_samples = config.sample_count * len(config.tasks)
        if len(samples[condition.name]) != expected_samples:
            raise ValueError(
                f"{condition.name} contains {len(samples[condition.name])} samples, expected {expected_samples}"
            )
        _validate_sample_rows(condition.name, samples[condition.name])
    if "full" not in results:
        raise ValueError("E2 analysis requires the full condition")
    rows: list[dict[str, Any]] = []
    for condition_name, result in results.items():
        for task, task_result in result["tasks"].items():
            score = float(task_result["primary_score"])
            full_score = float(results["full"]["tasks"][task]["primary_score"])
            if not math.isfinite(score) or not math.isfinite(full_score):
                raise ValueError(f"non-finite E2 score for {condition_name}/{task}")
            baseline = LOWRES_BASELINE.get(condition_name)
            gain = None
            if baseline in results:
                gain = score - float(results[baseline]["tasks"][task]["primary_score"])
            retention = _retention(samples["full"], samples[condition_name], task)
            agreement = _token_agreement(samples["full"], samples[condition_name], task)
            first_agreement = _first_token_agreement(samples["full"], samples[condition_name], task)
            timing = _mean_timing(samples[condition_name], task)
            pooling_gains = _native_pooling_gains(results, condition_name, task, score)
            rows.append(
                {
                    "condition": condition_name,
                    "task": task,
                    "score": score,
                    "delta_full": score - full_score,
                    "lowres_baseline": baseline,
                    "gain_lowres": gain,
                    "full_correct_retention": retention,
                    "token_agreement": agreement,
                    "first_token_agreement": first_agreement,
                    **pooling_gains,
                    **timing,
                }
            )
        rows.append(
            {
                "condition": condition_name,
                "task": "macro_average",
                "score": float(result["macro_average"]),
                "delta_full": float(result["macro_average"]) - float(results["full"]["macro_average"]),
                "lowres_baseline": LOWRES_BASELINE.get(condition_name),
                "gain_lowres": (
                    float(result["macro_average"]) - float(results[LOWRES_BASELINE[condition_name]]["macro_average"])
                    if LOWRES_BASELINE.get(condition_name) in results else None
                ),
                "full_correct_retention": None,
                "token_agreement": None,
                "first_token_agreement": None,
                **_native_pooling_gains(
                    results,
                    condition_name,
                    None,
                    float(result["macro_average"]),
                ),
                "prefill_seconds": None,
                "decode_seconds": None,
                "native_prefill_seconds": None,
                "native_bank_tokens": None,
                "total_seconds": None,
            }
        )
    (root / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _sample_map(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            rows[row["sample_id"]] = row
    return rows


def _retention(full: dict[str, Any], current: dict[str, Any], task: str) -> float | None:
    correct = [key for key, row in full.items() if row["task"] == task and _correct(row)]
    if not correct:
        return None
    return sum(_correct(current[key]) for key in correct) / len(correct)


def _correct(row: dict[str, Any]) -> bool:
    metrics = row["metrics"]
    value = metrics.get("relaxed_overall", metrics.get("exact_match", 0))
    return float(value) > 0


def _token_agreement(full: dict[str, Any], current: dict[str, Any], task: str) -> float | None:
    matches = total = 0
    for key, full_row in full.items():
        if full_row["task"] != task or key not in current:
            continue
        left, right = full_row["generated_token_ids"], current[key]["generated_token_ids"]
        length = min(len(left), len(right))
        matches += sum(left[index] == right[index] for index in range(length))
        total += max(len(left), len(right))
    return matches / total if total else None


def _first_token_agreement(full: dict[str, Any], current: dict[str, Any], task: str) -> float | None:
    values = []
    for key, full_row in full.items():
        if full_row["task"] != task or key not in current:
            continue
        left, right = full_row["generated_token_ids"], current[key]["generated_token_ids"]
        if left and right:
            values.append(left[0] == right[0])
    return sum(values) / len(values) if values else None


def _mean_timing(samples: dict[str, Any], task: str) -> dict[str, float | None]:
    rows = [row for row in samples.values() if row["task"] == task]
    result = {}
    for name in (
        "prefill_seconds", "decode_seconds", "native_prefill_seconds",
        "native_bank_tokens", "total_seconds",
    ):
        values = [float(row[name]) for row in rows if row.get(name) is not None]
        result[name] = sum(values) / len(values) if values else None
    return result


def _native_pooling_gains(
    results: dict[str, Any],
    condition: str,
    task: str | None,
    score: float,
) -> dict[str, float | None]:
    names = NATIVE_POOLING_BASELINES.get(condition, ())
    labels = ("gain_kv_pooling", "gain_hidden_pooling", "gain_postrope_pooling")
    gains: dict[str, float | None] = {label: None for label in labels}
    for label, baseline in zip(labels, names):
        if baseline not in results:
            continue
        baseline_score = (
            float(results[baseline]["macro_average"])
            if task is None
            else float(results[baseline]["tasks"][task]["primary_score"])
        )
        gains[label] = score - baseline_score
    return gains


def _validate_sample_rows(condition: str, samples: dict[str, dict[str, Any]]) -> None:
    for sample_id, row in samples.items():
        for name in ("prefill_seconds", "decode_seconds", "native_prefill_seconds", "total_seconds"):
            value = float(row.get(name, 0) or 0)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {name} for {condition}/{sample_id}")
        if condition in {"lowres_2", "lowres_4"} and int(row.get("token_count_delta", 0)) != 0:
            raise ValueError(f"{condition}/{sample_id} does not exactly match pooled token count")
