from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .config import E1Config
from .data import read_jsonl


def build_report(config: E1Config, model_name: str) -> Path:
    output_dir = config.output_dir / model_name
    metrics = _read_json(output_dir / "head_metrics.json")
    metadata = _read_json(output_dir / "probe_metadata.json")
    selection = _read_json(output_dir / "head_selection_top8.json")
    assets = output_dir / "report_assets"
    assets.mkdir(parents=True, exist_ok=True)
    basic_heatmap = assets / "layer_head_basic_score.png"
    gaze_heatmap = assets / "layer_head_gaze_score.png"
    _metric_heatmap(metrics, "basic_score", basic_heatmap)
    _metric_heatmap(metrics, "calibrated_gaze_score", gaze_heatmap, signed=True)
    gaze_images = _gaze_matrix_images(output_dir, selection, assets)
    trace_images = _trace_images(config, output_dir, selection, assets)
    rows = []
    metrics_by_key = {(row["layer"], row["head"]): row for row in metrics}
    for item in selection["selected_heads"]:
        row = metrics_by_key[(item["layer"], item["head"])]
        rows.append(
            "<tr>"
            f"<td>{row['layer']}</td><td>{row['head']}</td>"
            f"<td>{html.escape(row['hybrid_class'])}</td>"
            f"<td>{row['visual_mass']:.4f}</td><td>{row['concentration']:.4f}</td>"
            f"<td>{row['coverage']:.4f}</td><td>{row['persistence']:.4f}</td>"
            f"<td>{row['calibrated_gaze_score']:.5f}</td></tr>"
        )
    galleries = []
    for title, paths in (("Gaze matrices", gaze_images), ("Representative attention traces", trace_images)):
        if paths:
            images = "".join(
                f'<figure><img src="report_assets/{html.escape(path.name)}"><figcaption>{html.escape(path.stem)}</figcaption></figure>'
                for path in paths
            )
            galleries.append(f"<h2>{title}</h2><div class='gallery'>{images}</div>")
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>E1 report: {html.escape(model_name)}</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:1200px;margin:32px auto;padding:0 20px;color:#1f2328}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #d0d7de;padding:6px;text-align:right}}
th:nth-child(-n+3),td:nth-child(-n+3){{text-align:left}}img{{max-width:100%;border:1px solid #d0d7de}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px}}figure{{margin:0}}figcaption{{font-size:12px}}
</style></head><body>
<h1>E1 visual routing Head report</h1>
<p><b>Model:</b> {html.escape(str(metadata.get('model', model_name)))}</p>
<p><b>Full-attention layers:</b> {html.escape(str(metadata['full_attention_layers']))}</p>
<p><b>Samples:</b> {html.escape(str(metadata['sample_counts']))}</p>
<h2>Default Top-8 selection</h2>
<table><thead><tr><th>Layer</th><th>Head</th><th>Hybrid class</th><th>Visual mass</th>
<th>Concentration</th><th>Coverage</th><th>Persistence</th><th>Calibrated gaze</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>Layer × Head overview</h2><h3>Visual mass × concentration</h3>
<img src="report_assets/{basic_heatmap.name}"><h3>Calibrated GazeScore</h3>
<img src="report_assets/{gaze_heatmap.name}">
{''.join(galleries)}
</body></html>"""
    report_path = output_dir / "report.html"
    report_path.write_text(document, encoding="utf-8")
    return report_path


def _metric_heatmap(
    rows: list[dict[str, Any]], metric: str, path: Path, *, signed: bool = False
) -> None:
    layers = sorted({int(row["layer"]) for row in rows})
    heads = sorted({int(row["head"]) for row in rows})
    values = [float(row[metric]) for row in rows]
    scale = max(max(abs(value) for value in values), 1e-12) if signed else max(max(values), 1e-12)
    cell = 14
    margin_x, margin_y = 48, 30
    image = Image.new("RGB", (margin_x + cell * len(heads), margin_y + cell * len(layers)), "white")
    draw = ImageDraw.Draw(image)
    lookup = {(int(row["layer"]), int(row["head"])): float(row[metric]) for row in rows}
    for row_index, layer in enumerate(layers):
        draw.text((2, margin_y + row_index * cell), str(layer), fill="black")
        for column_index, head in enumerate(heads):
            value = lookup.get((layer, head), 0.0)
            color = _signed_color(value / scale) if signed else _positive_color(value / scale)
            x, y = margin_x + column_index * cell, margin_y + row_index * cell
            draw.rectangle((x, y, x + cell - 1, y + cell - 1), fill=color)
    draw.text((2, 2), "layer", fill="black")
    draw.text((margin_x, 2), f"head →  ({metric})", fill="black")
    image.save(path)


def _gaze_matrix_images(output_dir: Path, selection: dict[str, Any], assets: Path) -> list[Path]:
    gaze = _read_json(output_dir / "gaze_metrics.json")
    lookup = {(row["layer"], row["head"]): row for row in gaze}
    paths = []
    for item in selection["selected_heads"]:
        key = (item["layer"], item["head"])
        row = lookup.get(key)
        if row is None:
            continue
        path = assets / f"gaze_l{key[0]}_h{key[1]}.png"
        _small_matrix(row["matrix"], path)
        paths.append(path)
    return paths


def _small_matrix(matrix: list[list[float]], path: Path) -> None:
    size, cell, margin = len(matrix), 28, 30
    maximum = max(max(row) for row in matrix) or 1.0
    image = Image.new("RGB", (margin + size * cell, margin + size * cell), "white")
    draw = ImageDraw.Draw(image)
    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            draw.rectangle(
                (margin + x * cell, margin + y * cell, margin + (x + 1) * cell - 1, margin + (y + 1) * cell - 1),
                fill=_positive_color(value / maximum),
            )
    draw.text((2, 2), "query ↓ / attended panel →", fill="black")
    image.save(path)


def _trace_images(
    config: E1Config, output_dir: Path, selection: dict[str, Any], assets: Path
) -> list[Path]:
    trace_path = output_dir / "visualization" / "attention_traces.jsonl"
    if not trace_path.exists():
        return []
    natural_lookup = {row["id"]: row for row in read_jsonl(config.data_dir / "natural.jsonl")}
    controlled_lookup = {row["id"]: row for row in read_jsonl(config.data_dir / "controlled.jsonl")}
    selected = {(item["layer"], item["head"]) for item in selection["selected_heads"]}
    records = list(read_jsonl(trace_path))
    paths: list[Path] = []
    per_kind = {"trace": 0, "gaze": 0}
    for record in records:
        kind = record["kind"]
        limit = 3 if kind == "trace" else 2
        if kind not in per_kind or per_kind[kind] >= limit:
            continue
        if kind == "trace":
            source = natural_lookup.get(record["sample_id"])
        else:
            base_id = record["sample_id"].split("-panel-")[0]
            source = controlled_lookup.get(base_id)
        if source is None:
            continue
        source_image = Image.open(config.data_dir / source["image"]).convert("RGB")
        series: dict[tuple[int, int], list[list[float]]] = {key: [] for key in selected}
        for step in record["steps"]:
            for key_text, values in step["heads"].items():
                layer, head = (int(value) for value in key_text.split(":"))
                if (layer, head) in series:
                    series[(layer, head)].append(values["distribution"])
        for (layer, head), distributions in series.items():
            if not distributions:
                continue
            path = assets / f"trace_{_safe(record['sample_id'])}_l{layer}_h{head}.png"
            _attention_overlay(source_image, record["grid"], distributions, path)
            paths.append(path)
        per_kind[kind] += 1
    return paths


def _attention_overlay(
    image: Image.Image, grid: list[int], distributions: list[list[float]], path: Path
) -> None:
    rows, columns = (int(value) for value in grid)
    mean = [sum(values) / len(distributions) for values in zip(*distributions)]
    maximum = max(mean) or 1.0
    heat = Image.new("RGBA", (columns, rows))
    heat.putdata([(255, 30, 0, int(220 * value / maximum)) for value in mean])
    heat = heat.resize(image.size, Image.Resampling.BILINEAR)
    canvas = Image.alpha_composite(image.convert("RGBA"), heat)
    draw = ImageDraw.Draw(canvas)
    points = []
    for distribution in distributions:
        mass = sum(distribution) or 1.0
        center_x = sum((index % columns + 0.5) * value for index, value in enumerate(distribution)) / mass
        center_y = sum((index // columns + 0.5) * value for index, value in enumerate(distribution)) / mass
        points.append((center_x / columns * image.width, center_y / rows * image.height))
    if len(points) > 1:
        draw.line(points, fill=(0, 255, 255, 255), width=max(2, image.width // 256))
    for point in points:
        radius = max(2, image.width // 200)
        draw.ellipse((point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius), fill="cyan")
    canvas.convert("RGB").save(path)


def _positive_color(value: float) -> tuple[int, int, int]:
    value = min(max(value, 0.0), 1.0)
    return (255, int(255 * (1 - value)), int(255 * (1 - value)))


def _signed_color(value: float) -> tuple[int, int, int]:
    value = min(max(value, -1.0), 1.0)
    if value >= 0:
        return (255, int(255 * (1 - value)), int(255 * (1 - value)))
    return (int(255 * (1 + value)), int(255 * (1 + value)), 255)


def _safe(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))
