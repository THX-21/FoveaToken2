from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from PIL import Image


def annotation(doc: dict[str, Any]) -> dict[str, Any]:
    value = doc.get("annotations", doc)
    if not isinstance(value, dict):
        raise ValueError("FineRS row has no annotations object")
    return value


def answerable(doc: dict[str, Any]) -> bool:
    return bool(str(annotation(doc).get("A", "")).strip())


def finers_doc_to_visual(doc: dict[str, Any]) -> list[Image.Image]:
    ann = annotation(doc)
    path = Path(str(ann.get("image_path", "")))
    root = Path(
        os.getenv("FINERS4K_IMAGE_ROOT", "data/e4/datasets/finers4k/images")
    ).expanduser()
    if not path.is_file():
        candidates = (root / path, root / path.name, root / "all_images" / path.name)
        path = next((candidate for candidate in candidates if candidate.is_file()), path)
    if not path.is_file():
        raise FileNotFoundError(
            f"FineRS image not found: {path}; set FINERS4K_IMAGE_ROOT after extracting all_images.zip"
        )
    return [Image.open(path).convert("RGB")]


def _options(value: Any) -> str:
    if isinstance(value, dict):
        return "\n".join(f"{key}. {item}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return "\n".join(f"{letters[index]}. {item}" for index, item in enumerate(value))
    return str(value).strip()


def finers_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs=None) -> str:
    ann = annotation(doc)
    question = str(ann.get("Q", "")).strip()
    options = _options(ann.get("options", ""))
    suffix = "\nAnswer with the option letter only." if options else "\nAnswer briefly."
    return f"{question}{chr(10) + options if options else ''}{suffix}"


def finers_doc_to_target(doc: dict[str, Any]) -> str:
    return str(annotation(doc).get("A", "")).strip()


def normalize(value: Any) -> str:
    text = str(value).strip().lower()
    tagged = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL)
    if tagged:
        text = tagged[-1]
    plain = normalize_plain(text)
    if len(plain) == 1 and plain.isalpha():
        return plain
    match = re.search(r"(?:final\s+)?answer\s*(?:is|:)?\s*\(?([a-z])\)?(?:[.\s]|$)", text)
    if match:
        return match.group(1)
    return normalize_plain(text)


def normalize_plain(value: Any) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value).lower()).split())


def finers_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    return {
        "finers_qa_accuracy": float(normalize(results[0]) == normalize(finers_doc_to_target(doc)))
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
