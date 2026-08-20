from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .conditions import Suite
from .config import E4Config

COLUMNS = (
    "suite",
    "condition",
    "task",
    "score",
    "baseline",
    "delta_baseline",
    "control_correct_retention",
    "token_agreement",
    "first_token_agreement",
    "mean_first_divergence",
    "prefill_active_tokens",
    "active_tokens",
    "configured_compression_ratio",
    "achieved_compression_ratio",
    "token_retention_ratio",
    "budget_relative_error",
    "max_lowres_aspect_log_error",
    "compression_ratio",
    "scale_1_fraction",
    "scale_4_fraction",
    "scale_16_fraction",
    "scale_64_fraction",
    "roi_fine_gain",
    "route_event_count",
    "route_swap_count",
    "mean_front_jaccard",
    "prefill_seconds",
    "decode_seconds",
    "native_prefill_seconds",
    "total_seconds",
)


def build_report(
    config: E4Config,
    model_name: str,
    *,
    suites: list[Suite] | None = None,
) -> Path:
    root = config.output_dir / model_name
    summary_path = root / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("run E4 analyze before report")
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    if suites:
        rows = [row for row in rows if row["suite"] in suites]
    headers = "".join(f"<th>{html.escape(name)}</th>" for name in COLUMNS)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(_format(row.get(name)))}</td>" for name in COLUMNS)
        + "</tr>"
        for row in rows
    )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>E4 report: {html.escape(model_name)}</title>
<style>body{{font-family:system-ui,sans-serif;margin:2rem;color:#222}}table{{border-collapse:collapse;font-size:12px}}
th,td{{border:1px solid #ccc;padding:5px 8px}}th{{background:#eee;position:sticky;top:0}}</style>
</head><body><h1>E4 dynamic native-multiscale report</h1>
<p>VisualProbe is deterministic Acc@1. FineRS-QA excludes segmentation-only rows. Native conditions retain the full prompt cache and auxiliary banks.</p>
<table><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table></body></html>"""
    path = root / "report.html"
    path.write_text(document, encoding="utf-8")
    return path


def _format(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.5f}"
    return str(value)
