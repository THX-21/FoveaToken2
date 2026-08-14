from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .config import E3Config


_COLUMNS = (
    "condition",
    "task",
    "control_condition",
    "score",
    "delta_control",
    "control_correct_retention",
    "token_agreement",
    "first_token_agreement",
    "empty_answer_rate",
    "repetitive_answer_rate",
    "format_compliance_rate",
    "mean_analyze_words",
    "analyze_over_limit_rate",
    "visual_tokens",
    "active_tokens",
    "compression_ratio",
    "anchor_position_min",
    "anchor_position_max",
    "prefill_seconds",
    "decode_seconds",
    "native_prefill_seconds",
    "total_seconds",
)


def build_report(config: E3Config, model_name: str) -> Path:
    root = config.output_dir / model_name
    summary_path = root / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("run E3 analyze before report")
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    headers = "".join(f"<th>{html.escape(name)}</th>" for name in _COLUMNS)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(_format(row.get(name)))}</td>" for name in _COLUMNS)
        + "</tr>"
        for row in rows
    )
    regressions = _regressions(root)
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>E3 report: {html.escape(model_name)}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem;color:#222}}table{{border-collapse:collapse;font-size:13px}}
th,td{{border:1px solid #ccc;padding:5px 8px}}th{{background:#eee;position:sticky;top:0}}
</style></head><body><h1>E3 Text-Anchor report</h1>
<p>Text-Anchor is decode-only. A first-token agreement below 1.0 fails analysis.</p>
<h2>Summary</h2><table><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table>
<h2>Representative control-correct regressions</h2>{regressions}
</body></html>"""
    path = root / "report.html"
    path.write_text(document, encoding="utf-8")
    return path


def _format(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.5f}"
    return str(value)


def _regressions(root: Path) -> str:
    from .conditions import PAIRS

    records = []
    for current_name, control_name in PAIRS.items():
        control = _sample_map(root / control_name / "samples.jsonl")
        current = _sample_map(root / current_name / "samples.jsonl")
        for sample_id, baseline in control.items():
            row = current[sample_id]
            if _correct(baseline) and not _correct(row):
                records.append(
                    (
                        current_name,
                        row["task"],
                        sample_id,
                        baseline.get("prediction", ""),
                        row.get("prediction", ""),
                    )
                )
                if len(records) == 20:
                    break
        if len(records) == 20:
            break
    if not records:
        return "<p>No control-correct regression was found.</p>"
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>"
        for row in records
    )
    return (
        "<table><thead><tr><th>condition</th><th>task</th><th>sample</th>"
        "<th>control</th><th>text-anchor</th></tr></thead><tbody>"
        f"{body}</tbody></table>"
    )


def _sample_map(path: Path) -> dict[str, dict[str, Any]]:
    return {
        row["sample_id"]: row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        )
    }


def _correct(row: dict[str, Any]) -> bool:
    metrics = row["metrics"]
    return float(metrics.get("relaxed_overall", metrics.get("exact_match", 0))) > 0
