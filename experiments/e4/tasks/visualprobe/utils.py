from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from PIL import Image


def _root() -> Path:
    value = os.getenv("VISUALPROBE_ROOT")
    return Path(value).expanduser() if value else Path("data/e4/visualprobe")


def visualprobe_doc_to_visual(doc: dict[str, Any]) -> list[Image.Image]:
    values = doc.get("images")
    if not isinstance(values, (list, tuple)) or len(values) != 1:
        raise ValueError("VisualProbe requires exactly one image path")
    path = Path(str(values[0]))
    if not path.is_file():
        candidates = (_root() / path, _root() / path.name)
        path = next((candidate for candidate in candidates if candidate.is_file()), path)
    if not path.is_file():
        raise FileNotFoundError(
            f"VisualProbe image not found: {path}; set VISUALPROBE_ROOT to the dataset snapshot"
        )
    return [Image.open(path).convert("RGB")]


def visualprobe_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs=None) -> str:
    return str(doc["problem"]).replace("<image>", "").strip()


def visualprobe_doc_to_target(doc: dict[str, Any]) -> str:
    return str(doc["solution"])


def normalize_answer(value: Any) -> str:
    text = str(value).strip().lower()
    tagged = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL)
    if tagged:
        text = tagged[-1]
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def visualprobe_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    prediction = normalize_answer(results[0])
    target = normalize_answer(doc["solution"])
    return {"visualprobe_accuracy": float(prediction == target)}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
