from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .config import E2Config
from .runner import run_name


def build_report(config: E2Config, model_name: str, *, thinking: bool = False) -> Path:
    root = config.output_dir / run_name(model_name, thinking)
    summary_path = root / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("run E2 analyze before report")
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    assets = root / "report_assets"
    assets.mkdir(exist_ok=True)
    front_images = _render_fronts(root, assets)
    table = "".join(
        "<tr>" + "".join(f"<td>{html.escape(_format(row.get(key)))}</td>" for key in _COLUMNS) + "</tr>"
        for row in rows
    )
    headers = "".join(f"<th>{html.escape(name)}</th>" for name in _COLUMNS)
    figures = "".join(
        f'<figure><img src="report_assets/{html.escape(path.name)}"><figcaption>{html.escape(label)}</figcaption></figure>'
        for label, path in front_images
    )
    errors = _error_rows(root)
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>E2 report: {html.escape(model_name)}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem;color:#222}}table{{border-collapse:collapse;font-size:13px}}
th,td{{border:1px solid #ccc;padding:5px 8px}}th{{background:#eee;position:sticky;top:0}}
.figures{{display:flex;gap:1rem;flex-wrap:wrap}}figure{{margin:0}}img{{max-width:360px;border:1px solid #bbb}}
</style></head><body><h1>E2 coarse visual representation report</h1>
<p>Model: <code>{html.escape(run_name(model_name, thinking))}</code>. Scores use the original lmms-eval task metrics.</p>
<h2>Summary</h2><table><thead><tr>{headers}</tr></thead><tbody>{table}</tbody></table>
<h2>Representative regressions</h2>{errors}
<h2>Representative random fronts</h2><div class="figures">{figures}</div>
</body></html>"""
    path = root / "report.html"
    path.write_text(document, encoding="utf-8")
    return path


_COLUMNS = (
    "condition", "task", "score", "delta_full", "lowres_baseline", "gain_lowres",
    "gain_kv_pooling", "gain_hidden_pooling", "gain_postrope_pooling",
    "full_correct_retention", "token_agreement", "prefill_seconds", "decode_seconds",
    "native_prefill_seconds", "native_bank_tokens", "total_seconds", "first_token_agreement",
)


def _format(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.5f}"
    return str(value)


def _render_fronts(root: Path, assets: Path) -> list[tuple[str, Path]]:
    images = []
    for condition in (
        "random_fixed_kv_center", "random_fixed_native",
        "random_perstep_kv_center", "random_perstep_native",
    ):
        trace = root / condition / "front_traces.jsonl"
        if not trace.exists():
            continue
        records = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line]
        if not records:
            continue
        record = records[0]
        height, width = record["grid"]
        scale = max(2, 320 // max(height, width))
        image = Image.new("RGB", (width * scale, height * scale), "white")
        draw = ImageDraw.Draw(image)
        colors = {1: "#4daf4a", 2: "#377eb8", 4: "#e41a1c", 8: "#984ea3"}
        for y, x, size in record["nodes"]:
            draw.rectangle(
                (x * scale, y * scale, (x + size) * scale - 1, (y + size) * scale - 1),
                fill=colors[size], outline="black",
            )
        path = assets / f"{condition}_front.png"
        image.save(path)
        images.append((f"{condition}: {record['sample_id']}", path))
    return images


def _error_rows(root: Path) -> str:
    full_path = root / "full" / "samples.jsonl"
    if not full_path.exists():
        return "<p>No sample logs available.</p>"
    full = {row["sample_id"]: row for row in _read_jsonl(full_path)}
    rows = []
    for condition_dir in sorted(path for path in root.iterdir() if path.is_dir() and path.name != "full"):
        path = condition_dir / "samples.jsonl"
        if not path.exists():
            continue
        for row in _read_jsonl(path):
            baseline = full.get(row["sample_id"])
            if baseline is not None and _correct(baseline) and not _correct(row):
                rows.append((
                    condition_dir.name,
                    row["task"],
                    row["sample_id"],
                    baseline.get("prediction", ""),
                    row.get("prediction", ""),
                ))
                if len(rows) == 20:
                    break
        if len(rows) == 20:
            break
    if not rows:
        return "<p>No Full-correct regression was found in completed conditions.</p>"
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr><th>condition</th><th>task</th><th>sample</th><th>Full</th><th>condition</th></tr></thead><tbody>{body}</tbody></table>"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _correct(row: dict[str, Any]) -> bool:
    metrics = row["metrics"]
    return float(metrics.get("relaxed_overall", metrics.get("exact_match", 0))) > 0
