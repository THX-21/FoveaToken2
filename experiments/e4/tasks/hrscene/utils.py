from __future__ import annotations

import re
from typing import Any

from PIL import Image


def hrscene_doc_to_visual(doc: dict[str, Any]) -> list[Image.Image]:
    image = doc.get("image")
    if not isinstance(image, Image.Image):
        raise ValueError("HRScene image column must decode to a PIL image")
    return [image.convert("RGB")]


def hrscene_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs=None) -> str:
    return f"{str(doc['question']).strip()}\nAnswer briefly."


def hrscene_doc_to_target(doc: dict[str, Any]) -> str:
    return str(doc["answer"])


def normalize(value: Any) -> str:
    text = str(value).strip().lower()
    tagged = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL)
    if tagged:
        text = tagged[-1]
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def hrscene_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    return {"hrscene_accuracy": float(normalize(results[0]) == normalize(doc["answer"]))}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
