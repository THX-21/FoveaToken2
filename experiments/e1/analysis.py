from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

from .config import E1Config


SELECTION_SPECS = {4: 2, 8: 4, 16: 6}


def analyze(config: E1Config, model_name: str) -> list[dict[str, Any]]:
    output_dir = config.output_dir / model_name
    natural = _read_json(output_dir / "natural_metrics.json")
    gaze = _read_json(output_dir / "gaze_metrics.json")
    metadata = _read_json(output_dir / "probe_metadata.json")
    natural_by_head = {(row["layer"], row["head"]): row for row in natural}
    gaze_by_head = {(row["layer"], row["head"]): row for row in gaze}
    if not natural_by_head or not gaze_by_head:
        raise ValueError("probe output is incomplete; both natural and gaze metrics are required")
    expected_layers = set(metadata["full_attention_layers"])
    observed_layers = {layer for layer, _ in natural_by_head}
    if observed_layers != expected_layers:
        raise ValueError(
            f"natural probe layers {sorted(observed_layers)} do not match full-attention layers {sorted(expected_layers)}"
        )
    rows: list[dict[str, Any]] = []
    for key, natural_row in natural_by_head.items():
        gaze_row = gaze_by_head.get(key)
        if gaze_row is None:
            continue
        basic = float(natural_row["visual_mass"]) * float(natural_row["concentration"])
        row = {
            **natural_row,
            "basic_score": basic,
            "raw_gaze_score": float(gaze_row["raw_gaze_score"]),
            "null_gaze_score": float(gaze_row["null_gaze_score"]),
            "calibrated_gaze_score": float(gaze_row["calibrated_gaze_score"]),
        }
        _validate_metric_row(row)
        rows.append(row)
    rows.sort(key=lambda row: (row["basic_score"], row["calibrated_gaze_score"]), reverse=True)
    keep_count = max(1, math.ceil(len(rows) * config.basic_keep_fraction))
    candidates = rows[:keep_count]
    coverage_median = statistics.median(row["coverage"] for row in candidates)
    persistence_median = statistics.median(row["persistence"] for row in candidates)
    candidate_keys = {(row["layer"], row["head"]) for row in candidates}
    for row in rows:
        row["passed_basic_filter"] = (row["layer"], row["head"]) in candidate_keys
        row["hybrid_class"] = (
            "dynamic_gaze"
            if row["coverage"] >= coverage_median and row["persistence"] <= persistence_median
            else "stable_localizer"
        )
    ranked = sorted(
        (row for row in candidates if row["calibrated_gaze_score"] > 0),
        key=lambda row: (row["calibrated_gaze_score"], row["basic_score"]),
        reverse=True,
    )
    if not ranked:
        raise ValueError("no Head has a positive calibrated GazeScore")
    _write_json(output_dir / "head_metrics.json", rows)
    _write_csv(output_dir / "head_metrics.csv", rows)
    _write_json(
        output_dir / "hybridkv_classification.json",
        [
            {
                "layer": row["layer"],
                "head": row["head"],
                "coverage": row["coverage"],
                "persistence": row["persistence"],
                "class": row["hybrid_class"],
                "passed_basic_filter": row["passed_basic_filter"],
            }
            for row in rows
        ],
    )
    model_id = str(metadata.get("model", config.models[model_name].pretrained))
    selections = {}
    for requested, layer_cap in SELECTION_SPECS.items():
        selected = select_with_layer_cap(ranked, requested, layer_cap)
        payload = {
            "version": 1,
            "model": model_id,
            "selection_method": "visual_mass_x_concentration_then_calibrated_gaze",
            "recommended_aggregation": "mean",
            "requested_size": requested,
            "actual_size": len(selected),
            "max_layers": layer_cap,
            "selected_heads": [
                {
                    "layer": row["layer"],
                    "head": row["head"],
                    "hybrid_class": row["hybrid_class"],
                    "scores": {
                        "visual_mass": row["visual_mass"],
                        "concentration": row["concentration"],
                        "coverage": row["coverage"],
                        "persistence": row["persistence"],
                        "calibrated_gaze_score": row["calibrated_gaze_score"],
                    },
                }
                for row in selected
            ],
        }
        path = output_dir / f"head_selection_top{requested}.json"
        _write_json(path, payload)
        selections[requested] = payload
    _write_json(output_dir / "head_selection.json", selections[8])
    return rows


def select_with_layer_cap(
    ranked: list[dict[str, Any]], requested: int, layer_cap: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_layers: set[int] = set()
    for row in ranked:
        layer = int(row["layer"])
        if layer not in selected_layers and len(selected_layers) >= layer_cap:
            continue
        selected.append(row)
        selected_layers.add(layer)
        if len(selected) == requested:
            break
    return selected


def _validate_metric_row(row: dict[str, Any]) -> None:
    for name in ("visual_mass", "concentration", "coverage", "persistence"):
        value = float(row[name])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"invalid {name} for layer {row['layer']} head {row['head']}: {value}")
    for name in ("basic_score", "raw_gaze_score", "null_gaze_score", "calibrated_gaze_score"):
        if not math.isfinite(float(row[name])):
            raise ValueError(f"non-finite {name} for layer {row['layer']} head {row['head']}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "layer",
        "head",
        "samples",
        "steps",
        "visual_mass",
        "concentration",
        "basic_score",
        "coverage",
        "persistence",
        "hybrid_class",
        "raw_gaze_score",
        "null_gaze_score",
        "calibrated_gaze_score",
        "passed_basic_filter",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
