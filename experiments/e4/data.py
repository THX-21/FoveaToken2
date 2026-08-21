from __future__ import annotations

import json
import os
import random
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import E4Config

MANIFEST_VERSION = 2
VISUALPROBE_TASKS = ("visualprobe_easy", "visualprobe_medium", "visualprobe_hard")


def prepare_data(
    config: E4Config,
    tasks: dict[str, Any] | None = None,
    *,
    force: bool = False,
) -> Path:
    path = config.data_dir / "sample_manifest.json"
    if path.exists() and not force:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("version") == MANIFEST_VERSION:
            validate_manifest(config, existing)
            return path
    if tasks is None:
        from .evaluator import load_tasks

        tasks = load_tasks("qwen25", config.formal_tasks)
    formal: dict[str, list[dict[str, Any]]] = {}
    answerable: dict[str, list[int]] = {}
    for task_name in config.formal_tasks:
        documents = tasks[task_name].eval_docs
        indices = list(range(len(documents)))
        if task_name == "finers_qa":
            indices = [index for index in indices if _finers_answerable(documents[index])]
            if not indices:
                raise ValueError("FineRS-QA contains no answerable MVQA/OVQA rows")
        answerable[task_name] = indices
        formal[task_name] = [_record(task_name, index) for index in indices]

    mechanism: dict[str, list[dict[str, Any]]] = {}
    for task_name in ("vstar_bench", "finers_qa", "hrscene_testmini"):
        if task_name not in config.mechanism_tasks:
            continue
        selected = _stratified_sample(
            tasks[task_name].eval_docs,
            answerable[task_name],
            config.mechanism_count,
            config.seed,
            task_name,
        )
        mechanism[task_name] = [_record(task_name, index) for index in selected]

    visual_tasks = [name for name in VISUALPROBE_TASKS if name in config.mechanism_tasks]
    if visual_tasks:
        quotas = _largest_remainder(
            config.mechanism_count,
            [len(answerable[name]) for name in visual_tasks],
        )
        for task_name, quota in zip(visual_tasks, quotas):
            selected = _sample(
                answerable[task_name], quota, config.seed, f"visualprobe:{task_name}"
            )
            mechanism[task_name] = [_record(task_name, index) for index in selected]

    payload = {
        "version": MANIFEST_VERSION,
        "seed": config.seed,
        "formal": formal,
        "reasoning": mechanism,
        "compression": formal,
        "logical_counts": {
            "visualprobe": sum(len(mechanism.get(name, ())) for name in VISUALPROBE_TASKS),
            "vstar_bench": len(mechanism.get("vstar_bench", ())),
            "finers_qa": len(mechanism.get("finers_qa", ())),
            "hrscene_testmini": len(mechanism.get("hrscene_testmini", ())),
        },
    }
    validate_manifest(config, payload)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def prepare_external_assets(config: E4Config) -> dict[str, str]:
    """Validate and, when necessary, extract locally downloaded E4 image assets."""
    datasets_root = Path(
        os.getenv("E4_DATASETS_ROOT", str(config.data_dir / "datasets"))
    ).expanduser()
    visual_root = Path(
        os.getenv("VISUALPROBE_ROOT", str(datasets_root))
    ).expanduser()
    missing_visual: list[Path] = []
    for task_name in VISUALPROBE_TASKS:
        subset = datasets_root / task_name
        annotations = subset / "val.json"
        if not annotations.is_file():
            raise FileNotFoundError(f"missing VisualProbe annotations: {annotations}")
        for row in json.loads(annotations.read_text(encoding="utf-8")):
            relative = Path(str(row["images"][0]))
            candidates = (
                visual_root / relative,
                visual_root / task_name / "data" / relative.name,
                visual_root / relative.name,
            )
            if not any(candidate.is_file() for candidate in candidates):
                missing_visual.append(relative)
    if missing_visual:
        shown = ", ".join(str(path) for path in missing_visual[:5])
        raise FileNotFoundError(
            f"VisualProbe is missing {len(missing_visual)} referenced image(s): {shown}"
        )

    finers_root = Path(
        os.getenv(
            "FINERS4K_IMAGE_ROOT", str(datasets_root / "finers4k" / "images")
        )
    ).expanduser()
    finers_root.mkdir(parents=True, exist_ok=True)
    annotations = datasets_root / "finers4k" / "labels" / "all_annotations_final_test_v5.json"
    if not annotations.is_file():
        raise FileNotFoundError(f"missing FineRS annotations: {annotations}")
    payload = json.loads(annotations.read_text(encoding="utf-8"))
    referenced = [Path(str(row["image_path"])).name for row in payload["annotations"]]
    if referenced and not any(
        (finers_root / "all_images" / name).is_file() for name in referenced
    ):
        archive = datasets_root / "finers4k" / "all_images.zip"
        if not archive.is_file():
            raise FileNotFoundError(f"missing FineRS image archive: {archive}")
        _safe_extract(archive, finers_root)
    missing_finers = [
        name
        for name in referenced
        if not (finers_root / name).is_file()
        and not (finers_root / "all_images" / name).is_file()
    ]
    if missing_finers:
        shown = ", ".join(missing_finers[:5])
        raise FileNotFoundError(
            f"FineRS is missing {len(missing_finers)} referenced image(s): {shown}"
        )
    return {"visualprobe": str(visual_root), "finers4k": str(finers_root)}


def validate_manifest(config: E4Config, payload: dict[str, Any]) -> None:
    if payload.get("version") != MANIFEST_VERSION or payload.get("seed") != config.seed:
        raise ValueError("E4 manifest version or seed does not match config")
    formal = payload.get("formal")
    if not isinstance(formal, dict) or set(formal) != set(config.formal_tasks):
        raise ValueError("E4 formal manifest tasks do not match config")
    if payload.get("compression") != formal:
        raise ValueError("E4 compression manifest must reuse the formal sample indices")
    for suite in ("formal", "reasoning", "compression"):
        rows = payload.get(suite)
        if not isinstance(rows, dict):
            raise ValueError(f"E4 manifest is missing suite {suite!r}")
        ids = [record["sample_id"] for records in rows.values() for record in records]
        if len(ids) != len(set(ids)):
            raise ValueError(f"E4 {suite} manifest contains duplicate sample IDs")
    logical = payload.get("logical_counts", {})
    for name in ("visualprobe", "vstar_bench", "finers_qa", "hrscene_testmini"):
        configured = name == "visualprobe" and any(
            task in config.mechanism_tasks for task in VISUALPROBE_TASKS
        ) or name in config.mechanism_tasks
        expected = config.mechanism_count if configured else 0
        if int(logical.get(name, -1)) != expected:
            raise ValueError(
                f"E4 mechanism group {name!r} must contain {expected} samples"
            )


def suite_indices(config: E4Config, suite: str) -> dict[str, list[int]]:
    payload = json.loads((config.data_dir / "sample_manifest.json").read_text(encoding="utf-8"))
    validate_manifest(config, payload)
    if suite not in {"formal", "reasoning", "compression"}:
        raise ValueError(f"unsupported E4 suite: {suite}")
    return {
        task: [int(record["source_index"]) for record in records]
        for task, records in payload[suite].items()
    }


def _record(task_name: str, source_index: int) -> dict[str, Any]:
    return {"sample_id": f"{task_name}:{source_index}", "source_index": source_index}


def _finers_answerable(doc: Any) -> bool:
    annotation = doc.get("annotations", doc)
    return isinstance(annotation, dict) and bool(str(annotation.get("A", "")).strip())


def _sample(
    indices: list[int], count: int, seed: int, label: str
) -> list[int]:
    if len(indices) < count:
        raise ValueError(f"{label} has {len(indices)} rows but E4 requires {count}")
    rng = random.Random(f"{seed}:{label}")
    return sorted(rng.sample(indices, count))


def _stratified_sample(
    documents: Any,
    indices: list[int],
    count: int,
    seed: int,
    task_name: str,
) -> list[int]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        groups[_stratum(task_name, documents[index])].append(index)
    labels = sorted(groups)
    quotas = _largest_remainder(count, [len(groups[label]) for label in labels])
    selected: list[int] = []
    for label, quota in zip(labels, quotas):
        selected.extend(_sample(groups[label], quota, seed, f"{task_name}:{label}"))
    return sorted(selected)


def _stratum(task_name: str, doc: Any) -> str:
    if task_name == "finers_qa":
        annotation = doc.get("annotations", doc)
        return str(annotation.get("Q-type", annotation.get("attribute", "unknown")))
    for key in ("category", "data_source", "type", "dataset", "source"):
        value = doc.get(key) if hasattr(doc, "get") else None
        if value not in (None, ""):
            return str(value)
    identifier = str(doc.get("id", "unknown")) if hasattr(doc, "get") else "unknown"
    return identifier.split("_", 1)[0]


def _largest_remainder(total: int, sizes: list[int]) -> list[int]:
    if total <= 0 or not sizes or any(size <= 0 for size in sizes) or sum(sizes) < total:
        raise ValueError("invalid E4 stratified sample sizes")
    exact = [total * size / sum(sizes) for size in sizes]
    quotas = [min(size, int(value)) for size, value in zip(sizes, exact)]
    order = sorted(
        range(len(sizes)),
        key=lambda index: (exact[index] - quotas[index], sizes[index], -index),
        reverse=True,
    )
    remaining = total - sum(quotas)
    while remaining:
        progressed = False
        for index in order:
            if quotas[index] < sizes[index]:
                quotas[index] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise ValueError("cannot allocate E4 stratified sample quota")
    return quotas


def _safe_extract(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive) as stream:
        for member in stream.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"unsafe path in FineRS archive: {member.filename}")
        stream.extractall(destination)
