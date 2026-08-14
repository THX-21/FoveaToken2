from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from .conditions import CONDITIONS, PAIRS
from .config import E3Config
from .evaluator import PROMPT_VERSION


def analyze(config: E3Config, model_name: str) -> list[dict[str, Any]]:
    root = config.output_dir / model_name
    results: dict[str, Any] = {}
    samples: dict[str, dict[str, dict[str, Any]]] = {}
    expected = config.sample_count * len(config.tasks)
    for condition in CONDITIONS:
        result_path = root / condition.name / "results.json"
        sample_path = root / condition.name / "samples.jsonl"
        if not result_path.exists() or not sample_path.exists():
            raise FileNotFoundError(f"E3 condition {condition.name!r} is incomplete")
        results[condition.name] = json.loads(result_path.read_text(encoding="utf-8"))
        if results[condition.name].get("prompt_version") != PROMPT_VERSION:
            raise ValueError(f"{condition.name} uses an incompatible E3 prompt version")
        samples[condition.name] = _sample_map(sample_path)
        if len(samples[condition.name]) != expected:
            raise ValueError(
                f"{condition.name} contains {len(samples[condition.name])} samples, expected {expected}"
            )
        _validate_samples(condition.name, samples[condition.name], condition.text_anchor)

    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        result = results[condition.name]
        control_name = PAIRS.get(condition.name)
        for task in config.tasks:
            score = float(result["tasks"][task]["primary_score"])
            if not math.isfinite(score):
                raise ValueError(f"non-finite E3 score for {condition.name}/{task}")
            control_score = (
                float(results[control_name]["tasks"][task]["primary_score"])
                if control_name
                else None
            )
            pair_metrics: dict[str, float | None] = _pair_metrics(
                samples[control_name], samples[condition.name], task
            ) if control_name else _empty_pair_metrics()
            if control_name and pair_metrics["first_token_agreement"] != 1.0:
                raise ValueError(
                    f"prefill mismatch: {condition.name}/{task} first-token agreement is "
                    f"{pair_metrics['first_token_agreement']!r}, expected 1.0"
                )
            rows.append(
                {
                    "condition": condition.name,
                    "task": task,
                    "control_condition": control_name,
                    "score": score,
                    "delta_control": score - control_score if control_score is not None else None,
                    **pair_metrics,
                    **_diagnostics(samples[condition.name], task),
                }
            )
        macro = float(result["macro_average"])
        control_macro = (
            float(results[control_name]["macro_average"]) if control_name else None
        )
        pair_metrics = (
            _pair_metrics(samples[control_name], samples[condition.name], None)
            if control_name
            else _empty_pair_metrics()
        )
        if control_name and pair_metrics["first_token_agreement"] != 1.0:
            raise ValueError(
                f"prefill mismatch: {condition.name} first-token agreement is not 1.0"
            )
        rows.append(
            {
                "condition": condition.name,
                "task": "macro_average",
                "control_condition": control_name,
                "score": macro,
                "delta_control": macro - control_macro if control_macro is not None else None,
                **pair_metrics,
                **_diagnostics(samples[condition.name], None),
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


def _sample_map(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        sample_id = str(row["sample_id"])
        if sample_id in rows:
            raise ValueError(f"duplicate E3 sample ID {sample_id!r}")
        rows[sample_id] = row
    return rows


def _validate_samples(
    condition: str,
    samples: dict[str, dict[str, Any]],
    text_anchor: bool,
) -> None:
    for sample_id, row in samples.items():
        for name in (
            "prefill_seconds",
            "decode_seconds",
            "native_prefill_seconds",
            "total_seconds",
        ):
            value = float(row.get(name, 0) or 0)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {name} for {condition}/{sample_id}")
        tokens = row.get("generated_token_ids")
        if not isinstance(tokens, list) or not tokens:
            raise ValueError(f"missing generated tokens for {condition}/{sample_id}")
        word_count = int(row.get("analyze_word_count", -1))
        if word_count < 0:
            raise ValueError(f"missing Analyze word count for {condition}/{sample_id}")
        if text_anchor:
            minimum = float(row.get("anchor_position_min", math.nan))
            maximum = float(row.get("anchor_position_max", math.nan))
            if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum > maximum:
                raise ValueError(f"invalid anchor range for {condition}/{sample_id}")


def _pair_metrics(
    control: dict[str, dict[str, Any]],
    current: dict[str, dict[str, Any]],
    task: str | None,
) -> dict[str, float | None]:
    keys = [
        key for key, row in control.items()
        if task is None or row["task"] == task
    ]
    if any(key not in current for key in keys):
        raise ValueError("E3 pair contains mismatched sample IDs")
    correct = [key for key in keys if _correct(control[key])]
    retention = (
        sum(_correct(current[key]) for key in correct) / len(correct)
        if correct else None
    )
    matches = total = 0
    first = []
    for key in keys:
        left = control[key]["generated_token_ids"]
        right = current[key]["generated_token_ids"]
        shared = min(len(left), len(right))
        matches += sum(left[index] == right[index] for index in range(shared))
        total += max(len(left), len(right))
        if left and right:
            first.append(left[0] == right[0])
    return {
        "control_correct_retention": retention,
        "token_agreement": matches / total if total else None,
        "first_token_agreement": sum(first) / len(first) if first else None,
    }


def _empty_pair_metrics() -> dict[str, float | None]:
    return {
        "control_correct_retention": None,
        "token_agreement": None,
        "first_token_agreement": None,
    }


def _diagnostics(
    samples: dict[str, dict[str, Any]],
    task: str | None,
) -> dict[str, float | None]:
    rows = [row for row in samples.values() if task is None or row["task"] == task]
    timings = {}
    for name in (
        "prefill_seconds",
        "decode_seconds",
        "native_prefill_seconds",
        "total_seconds",
        "visual_tokens",
        "active_tokens",
        "compression_ratio",
    ):
        values = [float(row[name]) for row in rows if row.get(name) is not None]
        timings[name] = sum(values) / len(values) if values else None
    anchor_mins = [float(row["anchor_position_min"]) for row in rows if row.get("anchor_position_min") is not None]
    anchor_maxs = [float(row["anchor_position_max"]) for row in rows if row.get("anchor_position_max") is not None]
    return {
        **timings,
        "empty_answer_rate": sum(not str(row.get("prediction", "")).strip() for row in rows) / len(rows),
        "repetitive_answer_rate": sum(_repetitive(row["generated_token_ids"]) for row in rows) / len(rows),
        "invalid_sample_count": 0.0,
        "format_compliance_rate": sum(bool(row.get("format_compliant")) for row in rows) / len(rows),
        "mean_analyze_words": sum(int(row.get("analyze_word_count", 0)) for row in rows) / len(rows),
        "analyze_over_limit_rate": sum(
            int(row.get("analyze_word_count", 0)) > 200 for row in rows
        ) / len(rows),
        "anchor_position_min": min(anchor_mins) if anchor_mins else None,
        "anchor_position_max": max(anchor_maxs) if anchor_maxs else None,
    }


def _correct(row: dict[str, Any]) -> bool:
    metrics = row["metrics"]
    return float(metrics.get("relaxed_overall", metrics.get("exact_match", 0))) > 0


def _repetitive(tokens: list[int]) -> bool:
    if len(tokens) < 4:
        return False
    longest = run = 1
    for left, right in zip(tokens, tokens[1:]):
        run = run + 1 if left == right else 1
        longest = max(longest, run)
    return longest >= max(4, math.ceil(0.75 * len(tokens)))
