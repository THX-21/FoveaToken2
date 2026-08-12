from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from tokenfovea.router import SplitMergeRouter
from tokenfovea.topology import DeviceTreeTopology, VisualTokenForest


def aligned_size(height: int, width: int, factor: int) -> tuple[int, int]:
    return round(height / factor) * factor, round(width / factor) * factor


def leaf_target(image: Image.Image, grid: tuple[int, int], center: tuple[float, float]) -> torch.Tensor:
    height, width = grid
    pixels = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    rgb = torch.tensor(list(pixels.getdata()), dtype=torch.float32).reshape(height, width, 3) / 255.0
    luminance = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    edge = torch.zeros_like(luminance)
    edge[:, 1:] += (luminance[:, 1:] - luminance[:, :-1]).abs()
    edge[1:, :] += (luminance[1:, :] - luminance[:-1, :]).abs()
    edge /= edge.max().clamp_min(1e-12)
    y = (torch.arange(height) + 0.5) / height
    x = (torch.arange(width) + 0.5) / width
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    cy, cx = center
    focus = torch.exp(-((yy - cy).square() + (xx - cx).square()) / (2.0 * 0.07**2))
    scores = focus + 0.12 * edge
    return (scores / scores.sum()).flatten()


def front_metrics(
    forest: VisualTokenForest,
    active_ids: torch.Tensor,
    target: torch.Tensor,
    focus_leaf: int,
) -> dict:
    leaf_areas = torch.empty(forest.num_leaves)
    focus_area = None
    for node_id in active_ids.tolist():
        node = forest.node(node_id)
        leaf_areas[list(node.leaf_indices)] = node.valid_count
        if focus_leaf in node.leaf_indices:
            focus_area = node.valid_count
    return {
        "focus_covering_leaves": int(focus_area),
        "attention_weighted_covering_leaves": float((target * leaf_areas).sum()),
        "leaf_nodes": sum(forest.node(node_id).is_leaf for node_id in active_ids.tolist()),
    }


def render_front(
    image: Image.Image,
    forest: VisualTokenForest,
    active_ids: torch.Tensor,
    center: tuple[float, float] | None,
    label: str,
) -> Image.Image:
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result, "RGBA")
    grid_h, grid_w = forest.grids[0]
    for node_id in active_ids.tolist():
        node = forest.node(node_id)
        x0 = round(node.x0 * result.width / grid_w)
        x1 = round(node.x1 * result.width / grid_w)
        y0 = round(node.y0 * result.height / grid_h)
        y1 = round(node.y1 * result.height / grid_h)
        fine = node.valid_count == 1
        color = (255, 45, 45, 230) if fine else (40, 180, 255, 180)
        draw.rectangle((x0, y0, x1, y1), outline=color, width=2 if fine else 1)
    if center is not None:
        cy, cx = center
        radius = 12
        px, py = round(cx * result.width), round(cy * result.height)
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), outline=(0, 255, 80, 255), width=4)
    draw.rectangle((8, 8, 420, 38), fill=(0, 0, 0, 180))
    draw.text((16, 14), label, fill=(255, 255, 255, 255))
    return result


def run_mode(
    image: Image.Image,
    grid: tuple[int, int],
    budget: int,
    score_mode: str,
    output_dir: Path,
) -> tuple[list[dict], list[Image.Image]]:
    forest = VisualTokenForest.from_grids([grid])
    topology = DeviceTreeTopology.build(forest, torch.device("cpu"))
    initial = torch.tensor(sorted(forest.initial_front(budget)), dtype=torch.long)
    router = SplitMergeRouter(topology, initial, epsilon=0.02, max_swaps=8, score_mode=score_mode)
    fixed_budget = initial.numel()
    frames = [render_front(image, forest, router.active_ids(), None, f"{score_mode}: initial B={fixed_budget}")]
    records = []
    targets = (
        ("left facade", (0.63, 0.18)),
        ("central dome", (0.27, 0.50)),
        ("right facade", (0.63, 0.82)),
    )
    previous_front = set(router.active_ids().tolist())
    for step, (name, center) in enumerate(targets, start=1):
        target = leaf_target(image, grid, center)
        focus_row = min(grid[0] - 1, int(center[0] * grid[0]))
        focus_col = min(grid[1] - 1, int(center[1] * grid[1]))
        focus_leaf = focus_row * grid[1] + focus_col
        before = front_metrics(forest, router.active_ids(), target, focus_leaf)
        swaps = 0
        for _ in range(4):
            active_ids = router.active_ids()
            node_mass = topology.aggregate_leaves(target, density=False)
            active_mass = node_mass.index_select(0, active_ids)
            node_scores, _ = router.scores_from_active(active_ids, active_mass)
            swaps += int(router.step(node_scores))
            current = router.active_ids()
            assert current.numel() == fixed_budget
            forest.validate_front(current.tolist())
        active_ids = router.active_ids()
        after = front_metrics(forest, active_ids, target, focus_leaf)
        current_front = set(active_ids.tolist())
        records.append(
            {
                "step": step,
                "target": name,
                "center_yx": center,
                "swaps": swaps,
                "changed_nodes": len(previous_front.symmetric_difference(current_front)),
                "before": before,
                "after": after,
                "budget": int(fixed_budget),
            }
        )
        previous_front = current_front
        frame = render_front(image, forest, active_ids, center, f"{score_mode} step {step}: {name}, swaps={swaps}")
        frame.save(output_dir / f"{score_mode}_step{step}.png")
        frames.append(frame)
    if not all(record["swaps"] > 0 for record in records):
        raise AssertionError(f"{score_mode}: at least one attention move did not update the front")
    return records, frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test TokenFovea dynamic resolution on a real image")
    parser.add_argument("image", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dynamic_resolution_smoke"))
    parser.add_argument("--budget", type=int, default=128)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(args.image)
    resized_h, resized_w = aligned_size(image.height, image.width, factor=28)
    grid = (resized_h // 28, resized_w // 28)
    summary = {
        "image": str(args.image.resolve()),
        "original_size_wh": [image.width, image.height],
        "qwen25_aligned_size_wh": [resized_w, resized_h],
        "visual_grid_hw": list(grid),
        "requested_budget": args.budget,
        "modes": {},
    }
    rows = []
    for mode in ("mass", "density"):
        records, frames = run_mode(image, grid, args.budget, mode, args.output_dir)
        summary["modes"][mode] = records
        rows.append(frames)
    thumb_w = 420
    thumb_h = round(image.height * thumb_w / image.width)
    sheet = Image.new("RGB", (thumb_w * len(rows[0]), thumb_h * len(rows)), "white")
    for row, frames in enumerate(rows):
        for col, frame in enumerate(frames):
            sheet.paste(frame.resize((thumb_w, thumb_h)), (col * thumb_w, row * thumb_h))
    sheet.save(args.output_dir / "contact_sheet.png")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
