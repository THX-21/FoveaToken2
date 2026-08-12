from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageOps

from .config import E1Config


PANEL_NAMES = (
    "top-left",
    "top-center",
    "top-right",
    "middle-left",
    "center",
    "middle-right",
    "bottom-left",
    "bottom-center",
    "bottom-right",
)


def prepare_data(config: E1Config, *, force: bool = False) -> dict[str, int]:
    """Materialize deterministic natural-image and 3x3 controlled manifests."""

    natural_manifest = config.data_dir / "natural.jsonl"
    controlled_manifest = config.data_dir / "controlled.jsonl"
    if natural_manifest.exists() and controlled_manifest.exists() and not force:
        return {
            "natural": sum(1 for _ in read_jsonl(natural_manifest)),
            "controlled": sum(1 for _ in read_jsonl(controlled_manifest)),
        }

    from datasets import load_dataset  # type: ignore[import-untyped]

    natural_dir = config.data_dir / "images" / "natural"
    controlled_dir = config.data_dir / "images" / "controlled"
    natural_dir.mkdir(parents=True, exist_ok=True)
    controlled_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.seed)
    natural_records: list[dict[str, Any]] = []
    for source in config.natural_sources:
        dataset = load_dataset(
            "lmms-lab-encoder/LMMs-Eval-Lite",
            source.dataset_name,
            split="lite",
            token=True,
        )
        if source.count > len(dataset):
            raise ValueError(
                f"source {source.dataset_name} contains {len(dataset)} rows, fewer than requested {source.count}"
            )
        indices = rng.sample(range(len(dataset)), source.count)
        for ordinal, index in enumerate(indices):
            row = dataset[index]
            image = row["image"].convert("RGB")
            sample_id = f"{source.name}-{ordinal:04d}"
            relative_path = Path("images") / "natural" / f"{sample_id}.jpg"
            image.save(config.data_dir / relative_path, format="JPEG", quality=95)
            natural_records.append(
                {
                    "id": sample_id,
                    "source": source.name,
                    "source_index": index,
                    "image": str(relative_path),
                    "prompt": source.prompt,
                }
            )
    write_jsonl(natural_manifest, natural_records)

    shuffled = natural_records.copy()
    rng.shuffle(shuffled)
    controlled_records = []
    for composite_index in range(config.controlled_count):
        cells = [shuffled[(composite_index * 9 + cell) % len(shuffled)] for cell in range(9)]
        images = [Image.open(config.data_dir / cell["image"]).convert("RGB") for cell in cells]
        collage = make_collage(images)
        sample_id = f"grid-{composite_index:04d}"
        relative_path = Path("images") / "controlled" / f"{sample_id}.jpg"
        collage.save(config.data_dir / relative_path, format="JPEG", quality=95)
        controlled_records.append(
            {
                "id": sample_id,
                "image": str(relative_path),
                "cell_ids": [cell["id"] for cell in cells],
                "prompts": [
                    f"Focus only on the {name} panel of this 3 by 3 image grid. Briefly describe that panel."
                    for name in PANEL_NAMES
                ],
                "null_prompt": "Briefly describe this 3 by 3 image grid without prioritizing any panel.",
            }
        )
    write_jsonl(controlled_manifest, controlled_records)
    metadata = {
        "seed": config.seed,
        "natural_count": len(natural_records),
        "controlled_count": len(controlled_records),
        "sources": {source.name: source.count for source in config.natural_sources},
        "panel_names": list(PANEL_NAMES),
    }
    (config.data_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"natural": len(natural_records), "controlled": len(controlled_records)}


def make_collage(images: list[Image.Image], cell_size: int = 256, gap: int = 4) -> Image.Image:
    if len(images) != 9:
        raise ValueError("a controlled E1 collage requires exactly nine images")
    size = 3 * cell_size + 2 * gap
    canvas = Image.new("RGB", (size, size), "white")
    for index, image in enumerate(images):
        cell = ImageOps.fit(image.convert("RGB"), (cell_size, cell_size), method=Image.Resampling.LANCZOS)
        row, column = divmod(index, 3)
        canvas.paste(cell, (column * (cell_size + gap), row * (cell_size + gap)))
    return canvas


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
