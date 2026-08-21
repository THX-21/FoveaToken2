from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


LITE_REPO_IDS = {
    "lmms-lab/LMMs-Eval-Lite",
    "lmms-lab-encoder/LMMs-Eval-Lite",
}
GQA_REPO_IDS = {"lmms-lab/GQA", "lmms-lab-encoder/GQA"}

E4_REPOS = {
    "DreamMr/HR-Bench": ("hrbench_8k", "hrbench/hr_bench_8k.parquet"),
    "lmms-lab-encoder/vstar-bench": (
        "test",
        "vstar_bench/data/test-*.parquet",
    ),
    "Lin-Chen/MMStar": ("val", "mmstar/mmstar.parquet"),
    "lmms-lab-encoder/ChartQA": (
        "test",
        "chartqa/data/test-*.parquet",
    ),
    "lmms-lab-encoder/textvqa": (
        "validation",
        "textvqa/data/validation-*.parquet",
    ),
    "Wenliang04/HRScene": (
        "testmini",
        "hrscene/realworld_combined/train-*.parquet",
    ),
}

VISUALPROBE_REPOS = {
    "Mini-o3/VisualProbe_Easy": "visualprobe_easy",
    "Mini-o3/VisualProbe_Medium": "visualprobe_medium",
    "Mini-o3/VisualProbe_Hard": "visualprobe_hard",
}


def lite_root() -> Path:
    return Path(
        os.getenv("LMMS_EVAL_LITE_PATH", "lmms-lab-encoder/LMMs-Eval-Lite")
    ).expanduser().resolve()


def gqa_root() -> Path:
    return Path(os.getenv("LMMS_GQA_PATH", "lmms-lab-encoder/GQA")).expanduser().resolve()


def e4_root() -> Path:
    return Path(os.getenv("E4_DATASETS_ROOT", "data/e4/datasets")).expanduser().resolve()


def lite_parquet(config_name: str) -> Path:
    return _required_file(
        lite_root() / config_name / "lite-00000-of-00001.parquet",
        "LMMs-Eval-Lite",
    )


def load_lite_config(config_name: str):
    from datasets import load_dataset  # type: ignore[import-untyped]

    return load_dataset(
        "parquet",
        data_files={"lite": str(lite_parquet(config_name))},
        split="lite",
    )


@contextmanager
def use_local_lmms_datasets() -> Iterator[None]:
    """Redirect lmms-eval's Hub dataset calls to local ModelScope Parquet files."""

    import datasets  # type: ignore[import-untyped]

    original = datasets.load_dataset

    def local_load_dataset(path: str, name: str | None = None, *args: Any, **kwargs: Any):
        if path in LITE_REPO_IDS:
            if not name:
                raise ValueError("LMMs-Eval-Lite requires a dataset config name")
            split = kwargs.get("split")
            return original(
                "parquet",
                data_files={"lite": str(lite_parquet(name))},
                split=split,
            )
        if path in GQA_REPO_IDS and name == "testdev_balanced_images":
            parquet = _required_file(
                gqa_root()
                / "testdev_balanced_images"
                / "testdev-00000-of-00001.parquet",
                "GQA testdev_balanced_images",
            )
            split = kwargs.get("split")
            return original(
                "parquet",
                data_files={"testdev": str(parquet)},
                split=split,
            )
        if path == "initiacms/XLRS-Bench-lite":
            from datasets import DatasetDict, load_from_disk  # type: ignore[import-untyped]

            root = _required_dir(e4_root() / "xlrs_bench_lite", "XLRS-Bench-lite")
            dataset = load_from_disk(str(root))
            split = kwargs.get("split")
            if split is None:
                return dataset
            if isinstance(dataset, DatasetDict):
                return dataset[split]
            if split != "train":
                raise KeyError(f"XLRS-Bench-lite has no local split {split!r}")
            return dataset
        if path in E4_REPOS:
            split_name, pattern = E4_REPOS[path]
            files = sorted(e4_root().glob(pattern))
            if not files:
                raise FileNotFoundError(
                    f"missing local E4 dataset files: {e4_root() / pattern}"
                )
            dataset = original(
                "parquet",
                data_files={split_name: [str(file) for file in files]},
                split=kwargs.get("split"),
                batch_size=8,
            )
            if path == "Lin-Chen/MMStar":
                dataset = _cast_image_column(dataset, "image")
            return dataset
        if path in VISUALPROBE_REPOS:
            source = _required_file(
                e4_root() / VISUALPROBE_REPOS[path] / "val.json", path
            )
            return original(
                "json",
                data_files={"validation": str(source)},
                split=kwargs.get("split"),
            )
        if path == "Jiazuo98/Finers-4k-benchmark":
            from datasets import Dataset, DatasetDict  # type: ignore[import-untyped]

            source = _required_file(
                e4_root() / "finers4k" / "labels" / "all_annotations_final_test_v5.json",
                "FineRS-4K annotations",
            )
            payload = json.loads(source.read_text(encoding="utf-8"))
            dataset = Dataset.from_list(
                [{"annotations": annotation} for annotation in payload["annotations"]]
            )
            split = kwargs.get("split")
            if split is None:
                return DatasetDict({"test": dataset})
            if split != "test":
                raise KeyError(f"FineRS-4K has no local split {split!r}")
            return dataset
        return original(path, name, *args, **kwargs)

    datasets.load_dataset = local_load_dataset
    try:
        yield
    finally:
        datasets.load_dataset = original


def _required_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing local {label} file: {path}. "
            "Download the matching ModelScope dataset or set its LMMS_*_PATH variable."
        )
    return path


def _required_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"missing local {label} directory: {path}")
    return path


def _cast_image_column(dataset: Any, column: str):
    from datasets import DatasetDict, Image  # type: ignore[import-untyped]

    if isinstance(dataset, DatasetDict):
        return DatasetDict(
            {name: split.cast_column(column, Image()) for name, split in dataset.items()}
        )
    return dataset.cast_column(column, Image())
