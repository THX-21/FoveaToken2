from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


LITE_REPO_IDS = {
    "lmms-lab/LMMs-Eval-Lite",
    "lmms-lab-encoder/LMMs-Eval-Lite",
}
GQA_REPO_IDS = {"lmms-lab/GQA", "lmms-lab-encoder/GQA"}


def lite_root() -> Path:
    return Path(
        os.getenv("LMMS_EVAL_LITE_PATH", "lmms-lab-encoder/LMMs-Eval-Lite")
    ).expanduser().resolve()


def gqa_root() -> Path:
    return Path(os.getenv("LMMS_GQA_PATH", "lmms-lab-encoder/GQA")).expanduser().resolve()


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
