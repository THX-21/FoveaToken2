from __future__ import annotations

import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tokenfovea.integrations.qwen.common import (
    mrope_sections,
    qwen2_effective_rope,
    repeat_kv,
    rotate_full_key,
    rotate_partial_key,
)

from .metrics import gaze_statistics, hybrid_statistics, visual_statistics


def full_context_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    groups: int,
    scaling: float,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Post-softmax attention of the last query over the complete KV context."""

    query = query[..., -1:, :]
    repeated_key = repeat_kv(key, groups)
    logits = torch.matmul(query.float(), repeated_key.float().transpose(-2, -1)) * scaling
    mask = _last_query_mask(attention_mask, key.shape[-2], logits.device)
    if mask is not None:
        if mask.dtype == torch.bool:
            logits = logits.masked_fill(~mask, -torch.inf)
        else:
            logits = logits + mask.float()
    return torch.softmax(logits, dim=-1, dtype=torch.float32).squeeze(-2)


def _last_query_mask(mask: torch.Tensor | None, key_length: int, device: torch.device) -> torch.Tensor | None:
    if mask is None:
        return None
    mask = mask.to(device)
    if mask.shape[-1] < key_length:
        raise ValueError("attention mask is shorter than the accumulated E1 key cache")
    mask = mask[..., :key_length]
    if mask.ndim == 2:
        mask = mask[:, None, None, :]
        if mask.is_floating_point():
            return mask > 0
        return mask != 0
    mask = mask[..., -1:, :]
    if mask.dtype == torch.bool:
        return mask
    if not mask.is_floating_point():
        return mask != 0
    return mask


@dataclass(slots=True)
class ProbeHandle:
    hooks: list[Any]

    def remove(self) -> None:
        for hook in self.hooks:
            hook.remove()


class E1AttentionProbe:
    """Non-invasive last-query attention probe for Qwen vision-language models."""

    def __init__(
        self,
        model: torch.nn.Module,
        output_dir: str | Path,
        *,
        top_fraction: float = 0.05,
        trace_heads: set[tuple[int, int]] | None = None,
        trace_path: str | Path | None = None,
        checkpoint: bool = False,
    ) -> None:
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.top_fraction = top_fraction
        self.trace_heads = trace_heads or set()
        self.trace_path = Path(trace_path) if trace_path is not None else None
        self.checkpoint = checkpoint
        self.checkpoint_path = self.output_dir / "probe_checkpoint.json"
        self.model_type = str(getattr(model.config, "model_type", ""))
        if self.model_type not in {"qwen2_5_vl", "qwen3_5"}:
            raise ValueError(f"unsupported E1 model type: {self.model_type}")
        self.language_model: Any = model.model.language_model
        self.image_token_id = int(model.config.image_token_id)
        self.spatial_merge_size = int(model.config.vision_config.spatial_merge_size)
        self.layer_types = getattr(self.language_model.config, "layer_types", None) or [
            "full_attention"
        ] * len(self.language_model.layers)
        self.routed_layers = tuple(
            index for index, layer_type in enumerate(self.layer_types) if layer_type == "full_attention"
        )
        self._natural_totals: dict[tuple[int, int], dict[str, float]] = {}
        self._gaze_totals: dict[tuple[int, int], dict[str, Any]] = {}
        self._sample_count = {"natural": 0, "gaze": 0, "null": 0}
        self._completed_sample_ids: set[str] = set()
        if self.checkpoint:
            self._load_checkpoint()
        self._reset_sample()
        self.handle = self._install()

    def _reset_sample(self) -> None:
        self.sample_id: str | None = None
        self.sample_kind: str | None = None
        self.target_panel: int | None = None
        self.visual_positions: torch.Tensor | None = None
        self.grid: tuple[int, int] | None = None
        self.panel_ids: torch.Tensor | None = None
        self._keys: dict[int, torch.Tensor] = {}
        self._step_counts: dict[int, int] = {}
        self._topk_counts: dict[int, torch.Tensor] = {}
        self._sample_sums: dict[int, dict[str, torch.Tensor]] = {}
        self._last_region_mass: dict[int, torch.Tensor] = {}
        self._trace_steps: list[dict[str, Any]] = []

    def begin_sample(
        self,
        sample_id: str,
        kind: str,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        *,
        target_panel: int | None = None,
    ) -> None:
        if kind not in {"natural", "gaze", "null", "trace"}:
            raise ValueError(f"unsupported E1 sample kind: {kind}")
        self._reset_sample()
        self.sample_id = sample_id
        self.sample_kind = kind
        self.target_panel = target_panel
        ids = input_ids[0].detach().cpu()
        positions = torch.nonzero(ids == self.image_token_id, as_tuple=False).flatten()
        grids = image_grid_thw.detach().cpu().tolist()
        if len(grids) != 1:
            raise ValueError("E1 currently requires exactly one image per sample")
        temporal, height, width = (int(value) for value in grids[0])
        if temporal != 1:
            raise ValueError("E1 currently supports images only")
        if height % self.spatial_merge_size or width % self.spatial_merge_size:
            raise ValueError("vision grid is not divisible by spatial_merge_size")
        rows, columns = height // self.spatial_merge_size, width // self.spatial_merge_size
        if positions.numel() != rows * columns:
            raise ValueError(
                f"processor emitted {positions.numel()} visual tokens but grid describes {rows * columns}"
            )
        self.visual_positions = positions
        self.grid = (rows, columns)
        row = torch.arange(rows).repeat_interleave(columns)
        column = torch.arange(columns).repeat(rows)
        panel_row = torch.clamp(((row.float() + 0.5) * 3 / rows).long(), max=2)
        panel_column = torch.clamp(((column.float() + 0.5) * 3 / columns).long(), max=2)
        self.panel_ids = panel_row * 3 + panel_column

    def end_sample(self, generated_text: str = "") -> None:
        if self.sample_id is None or self.sample_kind is None:
            return
        sample_id = self.sample_id
        if self.sample_kind in {"natural", "trace"}:
            self._finish_natural()
        elif self.sample_kind in {"gaze", "null"}:
            self._finish_gaze()
        if self.trace_path is not None and self._trace_steps:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "sample_id": self.sample_id,
                "kind": self.sample_kind,
                "target_panel": self.target_panel,
                "grid": list(self.grid or ()),
                "generated_text": generated_text,
                "steps": self._trace_steps,
            }
            with self.trace_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self.checkpoint:
            self._completed_sample_ids.add(sample_id)
            self.save()
        self._reset_sample()

    def is_complete(self, sample_id: str) -> bool:
        return sample_id in self._completed_sample_ids

    def _finish_natural(self) -> None:
        for layer, sums in self._sample_sums.items():
            steps = self._step_counts[layer]
            hybrid = hybrid_statistics(self._topk_counts[layer], steps)
            visual_mass = sums["visual_mass"] / steps
            concentration = sums["concentration"] / steps
            for head in range(visual_mass.numel()):
                key = (layer, head)
                total = self._natural_totals.setdefault(
                    key,
                    {"samples": 0.0, "steps": 0.0, "visual_mass": 0.0, "concentration": 0.0,
                     "coverage": 0.0, "persistence": 0.0},
                )
                total["samples"] += 1
                total["steps"] += steps
                total["visual_mass"] += float(visual_mass[head])
                total["concentration"] += float(concentration[head])
                total["coverage"] += hybrid[head].coverage
                total["persistence"] += hybrid[head].persistence
        self._sample_count["natural"] += 1

    def _finish_gaze(self) -> None:
        assert self.sample_kind is not None
        for layer, region_mass in self._last_region_mass.items():
            for head in range(region_mass.shape[0]):
                key = (layer, head)
                total = self._gaze_totals.setdefault(
                    key,
                    {
                        "matrix": torch.zeros(9, 9, dtype=torch.float64),
                        "matrix_count": torch.zeros(9, dtype=torch.int64),
                        "null": torch.zeros(9, dtype=torch.float64),
                        "null_count": 0,
                    },
                )
                if self.sample_kind == "null":
                    total["null"] += region_mass[head].double()
                    total["null_count"] += 1
                else:
                    if self.target_panel is None or not 0 <= self.target_panel < 9:
                        raise ValueError("gaze sample requires target_panel in [0, 8]")
                    total["matrix"][self.target_panel] += region_mass[head].double()
                    total["matrix_count"][self.target_panel] += 1
        self._sample_count[self.sample_kind] += 1

    def summaries(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        natural = []
        for (layer, head), total in sorted(self._natural_totals.items()):
            samples = max(total["samples"], 1.0)
            natural.append(
                {
                    "layer": layer,
                    "head": head,
                    "samples": int(total["samples"]),
                    "steps": int(total["steps"]),
                    **{name: total[name] / samples for name in ("visual_mass", "concentration", "coverage", "persistence")},
                }
            )
        gaze = []
        for (layer, head), gaze_total in sorted(self._gaze_totals.items()):
            counts = gaze_total["matrix_count"].clamp_min(1).double()
            matrix = gaze_total["matrix"] / counts[:, None]
            null_count = max(int(gaze_total["null_count"]), 1)
            null = gaze_total["null"] / null_count
            scores = gaze_statistics(matrix.unsqueeze(0), null.unsqueeze(0))[0]
            gaze.append(
                {
                    "layer": layer,
                    "head": head,
                    "raw_gaze_score": scores.raw_score,
                    "null_gaze_score": scores.null_score,
                    "calibrated_gaze_score": scores.calibrated_score,
                    "matrix": matrix.tolist(),
                    "null_vector": null.tolist(),
                    "target_counts": gaze_total["matrix_count"].tolist(),
                    "null_count": int(gaze_total["null_count"]),
                }
            )
        metadata = {
            "model_type": self.model_type,
            "full_attention_layers": list(self.routed_layers),
            "sample_counts": self._sample_count,
        }
        return natural, gaze, metadata

    def save(self) -> None:
        natural, gaze, metadata = self.summaries()
        _write_json(self.output_dir / "natural_metrics.json", natural)
        _write_json(self.output_dir / "gaze_metrics.json", gaze)
        _write_json(self.output_dir / "probe_metadata.json", metadata)
        if self.checkpoint:
            _write_json(
                self.checkpoint_path,
                {
                    "completed_sample_ids": sorted(self._completed_sample_ids),
                    "natural_totals": _encode_totals(self._natural_totals),
                    "gaze_totals": _encode_gaze_totals(self._gaze_totals),
                    "sample_counts": self._sample_count,
                },
            )

    def _load_checkpoint(self) -> None:
        if not self.checkpoint_path.exists():
            return
        payload = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        self._completed_sample_ids = set(payload.get("completed_sample_ids", []))
        self._natural_totals = _decode_totals(payload.get("natural_totals", {}))
        self._gaze_totals = _decode_gaze_totals(payload.get("gaze_totals", {}))
        self._sample_count.update(payload.get("sample_counts", {}))

    def _install(self) -> ProbeHandle:
        hooks = []
        for layer_index in self.routed_layers:
            module = self.language_model.layers[layer_index].self_attn
            signature = inspect.signature(module.forward)

            def pre_hook(_module, args, kwargs, *, index=layer_index, sig=signature):
                if self.sample_id is None:
                    return
                arguments = _call_arguments(sig, args, kwargs)
                self._observe(index, _module, arguments)

            hooks.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
        return ProbeHandle(hooks)

    @torch.no_grad()
    def _observe(self, layer: int, module: Any, arguments: dict[str, Any]) -> None:
        hidden = arguments.get("hidden_states")
        position_embeddings = arguments.get("position_embeddings")
        if hidden is None or position_embeddings is None or self.visual_positions is None:
            return
        query, key = self._project(module, hidden, position_embeddings)
        cached = self._keys.get(layer)
        full_key = key.detach() if cached is None else torch.cat((cached, key.detach()), dim=-2)
        self._keys[layer] = full_key
        weights = full_context_attention(
            query,
            full_key,
            int(module.num_key_value_groups),
            float(module.scaling),
            arguments.get("attention_mask"),
        )[0]
        visual_index = self.visual_positions.to(weights.device)
        visual = weights.index_select(-1, visual_index)
        mass, concentration, distribution = visual_statistics(visual)
        step = self._step_counts.get(layer, 0)
        self._step_counts[layer] = step + 1
        sums = self._sample_sums.setdefault(
            layer,
            {"visual_mass": torch.zeros_like(mass, device="cpu"), "concentration": torch.zeros_like(mass, device="cpu")},
        )
        sums["visual_mass"] += mass.cpu()
        sums["concentration"] += concentration.cpu()
        top_count = max(1, math.ceil(self.top_fraction * distribution.shape[-1]))
        top_indices = distribution.topk(top_count, dim=-1).indices.cpu()
        counts = self._topk_counts.setdefault(
            layer, torch.zeros(distribution.shape[0], distribution.shape[-1], dtype=torch.int32)
        )
        counts.scatter_add_(1, top_indices, torch.ones_like(top_indices, dtype=counts.dtype))
        assert self.panel_ids is not None
        panel_ids = self.panel_ids.to(distribution.device)
        region_mass = torch.zeros(distribution.shape[0], 9, dtype=visual.dtype, device=visual.device)
        region_mass.scatter_add_(1, panel_ids.expand(distribution.shape[0], -1), visual)
        self._last_region_mass[layer] = region_mass.cpu()
        if self.trace_heads:
            selected = {}
            for selected_layer, head in sorted(self.trace_heads):
                if selected_layer == layer and head < distribution.shape[0]:
                    selected[f"{layer}:{head}"] = {
                        "visual_mass": float(mass[head]),
                        "concentration": float(concentration[head]),
                        "distribution": distribution[head].cpu().tolist(),
                    }
            if selected:
                self._trace_steps.append({"layer": layer, "step": step, "heads": selected})

    def _project(
        self,
        module: Any,
        hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, _ = hidden.shape
        if self.model_type == "qwen2_5_vl":
            query = module.q_proj(hidden).view(batch, length, -1, module.head_dim).transpose(1, 2)
            key = module.k_proj(hidden).view(batch, length, -1, module.head_dim).transpose(1, 2)
            cos, sin = qwen2_effective_rope(*position_embeddings, mrope_sections(module.config))
            return rotate_full_key(query, cos, sin), rotate_full_key(key, cos, sin)
        hidden_shape = (batch, length, -1, module.head_dim)
        query_and_gate = module.q_proj(hidden).view(batch, length, -1, module.head_dim * 2)
        query, _ = torch.chunk(query_and_gate, 2, dim=-1)
        query = module.q_norm(query.view(hidden_shape)).transpose(1, 2)
        key = module.k_norm(module.k_proj(hidden).view(hidden_shape)).transpose(1, 2)
        cos, sin = position_embeddings
        return rotate_partial_key(query, cos, sin), rotate_partial_key(key, cos, sin)


def _call_arguments(signature: inspect.Signature, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    arguments = dict(kwargs)
    bound = signature.bind_partial(*args, **kwargs)
    for name, value in bound.arguments.items():
        if signature.parameters[name].kind == inspect.Parameter.VAR_KEYWORD:
            arguments.update(value)
        else:
            arguments[name] = value
    return arguments


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def _encode_totals(totals: dict[tuple[int, int], dict[str, float]]) -> dict[str, dict[str, float]]:
    return {f"{layer}:{head}": values for (layer, head), values in totals.items()}


def _decode_totals(payload: dict[str, dict[str, float]]) -> dict[tuple[int, int], dict[str, float]]:
    return {tuple(map(int, key.split(":"))): values for key, values in payload.items()}


def _encode_gaze_totals(totals: dict[tuple[int, int], dict[str, Any]]) -> dict[str, dict[str, Any]]:
    encoded = {}
    for (layer, head), values in totals.items():
        encoded[f"{layer}:{head}"] = {
            "matrix": values["matrix"].tolist(),
            "matrix_count": values["matrix_count"].tolist(),
            "null": values["null"].tolist(),
            "null_count": values["null_count"],
        }
    return encoded


def _decode_gaze_totals(payload: dict[str, dict[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    decoded = {}
    for key, values in payload.items():
        decoded[tuple(map(int, key.split(":")))] = {
            "matrix": torch.tensor(values["matrix"], dtype=torch.float64),
            "matrix_count": torch.tensor(values["matrix_count"], dtype=torch.int64),
            "null": torch.tensor(values["null"], dtype=torch.float64),
            "null_count": values["null_count"],
        }
    return decoded


def checkpoint_sample_ids(paths: list[Path]) -> set[str]:
    completed: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        completed.update(str(value) for value in payload.get("completed_sample_ids", []))
    return completed


def merge_probe_checkpoints(paths: list[Path], destination: Path) -> int:
    """Merge disjoint rank-local E1 accumulators into one resumable checkpoint."""

    completed: set[str] = set()
    natural_totals: dict[tuple[int, int], dict[str, float]] = {}
    gaze_totals: dict[tuple[int, int], dict[str, Any]] = {}
    sample_counts = {"natural": 0, "gaze": 0, "null": 0}
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        path_ids = set(str(value) for value in payload.get("completed_sample_ids", []))
        overlap = completed.intersection(path_ids)
        if overlap:
            example = sorted(overlap)[0]
            raise ValueError(f"duplicate E1 sample {example!r} across rank checkpoints")
        completed.update(path_ids)
        for key, values in _decode_totals(payload.get("natural_totals", {})).items():
            target = natural_totals.setdefault(key, {})
            for name, value in values.items():
                target[name] = target.get(name, 0.0) + float(value)
        for key, values in _decode_gaze_totals(payload.get("gaze_totals", {})).items():
            target = gaze_totals.setdefault(
                key,
                {
                    "matrix": torch.zeros(9, 9, dtype=torch.float64),
                    "matrix_count": torch.zeros(9, dtype=torch.int64),
                    "null": torch.zeros(9, dtype=torch.float64),
                    "null_count": 0,
                },
            )
            target["matrix"] += values["matrix"]
            target["matrix_count"] += values["matrix_count"]
            target["null"] += values["null"]
            target["null_count"] += int(values["null_count"])
        for name in sample_counts:
            sample_counts[name] += int(payload.get("sample_counts", {}).get(name, 0))
    _write_json(
        destination,
        {
            "completed_sample_ids": sorted(completed),
            "natural_totals": _encode_totals(natural_totals),
            "gaze_totals": _encode_gaze_totals(gaze_totals),
            "sample_counts": sample_counts,
        },
    )
    return len(completed)
