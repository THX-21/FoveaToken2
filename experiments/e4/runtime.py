from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from tokenfovea.session import RouteEvent


def validate_head_selection(
    path: str | Path,
    expected_model: str,
    *,
    routed_layers: set[int] | None = None,
    heads_per_layer: dict[int, int] | None = None,
) -> dict[str, Any]:
    selection_path = Path(path)
    if not selection_path.is_file():
        raise FileNotFoundError(
            f"E4 Top-8 selection not found: {selection_path}. Run E1 analyze first."
        )
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    if payload.get("version") != 1 or payload.get("model") != expected_model:
        raise ValueError("E4 Top-8 selection version or model does not match the configured model")
    pairs = payload.get("selected_heads")
    if not isinstance(pairs, list) or len(pairs) != 8:
        raise ValueError("E4 requires exactly eight selected E1 heads")
    seen: set[tuple[int, int]] = set()
    for pair in pairs:
        layer, head = int(pair["layer"]), int(pair["head"])
        if layer < 0 or head < 0 or (layer, head) in seen:
            raise ValueError("E4 Top-8 selection contains invalid or duplicate heads")
        if routed_layers is not None and layer not in routed_layers:
            raise ValueError(f"selected layer {layer} is not a full-attention layer")
        if heads_per_layer is not None and head >= heads_per_layer[layer]:
            raise ValueError(f"selected head {head} is out of range for layer {layer}")
        seen.add((layer, head))
    return payload


class RouteTraceObserver:
    """Convert explicitly requested route events to compact JSON-safe records."""

    def __init__(self) -> None:
        self.sample_id = ""
        self.rows: list[dict[str, Any]] = []

    def begin_sample(self, sample_id: str) -> None:
        self.sample_id = sample_id
        self.rows = []

    def __call__(self, event: RouteEvent) -> None:
        before = sorted(int(value) for value in event.active_before.detach().cpu().tolist())
        after = sorted(int(value) for value in event.active_after.detach().cpu().tolist())
        scores = event.node_scores.detach().float().cpu()
        top_count = min(10, scores.numel())
        top_values, top_ids = torch.topk(scores, top_count)
        before_set, after_set = set(before), set(after)
        union = before_set | after_set
        self.rows.append(
            {
                "sample_id": self.sample_id,
                "phase": event.phase,
                "step": event.step,
                "updated": event.updated,
                "swaps": int(event.swaps.detach().cpu()),
                "active_before": [_node(event, node_id) for node_id in before],
                "active_after": [_node(event, node_id) for node_id in after],
                "front_hash": _front_hash(after),
                "front_jaccard": len(before_set & after_set) / len(union) if union else 1.0,
                "score_min": float(scores.min()) if scores.numel() else None,
                "score_max": float(scores.max()) if scores.numel() else None,
                "score_sum": float(scores.sum()),
                "top_scores": [
                    {"node_id": int(node_id), "score": float(value)}
                    for node_id, value in zip(top_ids.tolist(), top_values.tolist())
                ],
            }
        )

    def drain(self) -> list[dict[str, Any]]:
        rows, self.rows = self.rows, []
        return rows


def _node(event: RouteEvent, node_id: int) -> dict[str, int]:
    node = event.forest.node(node_id)
    return {
        "node_id": node_id,
        "image_index": node.image_index,
        "y0": node.y0,
        "x0": node.x0,
        "y1": node.y1,
        "x1": node.x1,
        "area_scale": node.valid_count,
    }


def _front_hash(node_ids: list[int]) -> str:
    value = ",".join(map(str, node_ids)).encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:16]


class ForwardTimer:
    """Measure native auxiliary, prompt-prefill, and decode forwards."""

    def __init__(self, model: Any, native_scale: Callable[[], int | None]):
        self.model = model
        self.native_scale = native_scale
        self.handles: list[Any] = []
        self.started = 0.0
        self.phase = ""
        self.reset()

    def reset(self) -> None:
        self.prefill_seconds = 0.0
        self.decode_seconds = 0.0
        self.native_prefill_seconds = 0.0

    def install(self) -> "ForwardTimer":
        self.handles = [
            self.model.register_forward_pre_hook(self._before, with_kwargs=True),
            self.model.register_forward_hook(self._after, with_kwargs=True),
        ]
        return self

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _before(self, _module: Any, _args: Any, kwargs: dict[str, Any]) -> None:
        _synchronize()
        self.started = time.perf_counter()
        if self.native_scale() is not None:
            self.phase = "native"
            return
        cache = kwargs.get("past_key_values")
        get_length = getattr(cache, "get_seq_length", None)
        length = int(get_length()) if callable(get_length) else 0
        self.phase = "decode" if length else "prefill"

    def _after(self, _module: Any, _args: Any, _kwargs: Any, _output: Any) -> None:
        _synchronize()
        elapsed = time.perf_counter() - self.started
        if self.phase == "native":
            self.native_prefill_seconds += elapsed
        elif self.phase == "decode":
            self.decode_seconds += elapsed
        else:
            self.prefill_seconds += elapsed

    def diagnostics(self) -> dict[str, float]:
        return {
            "prefill_seconds": self.prefill_seconds,
            "decode_seconds": self.decode_seconds,
            "native_prefill_seconds": self.native_prefill_seconds,
        }


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
