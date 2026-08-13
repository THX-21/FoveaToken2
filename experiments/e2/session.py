from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tokenfovea.integrations.qwen.common import rotate_full_key, rotate_partial_key  # type: ignore[import-untyped]

from .conditions import Condition
from .front import BlockFront, stable_seed

PositionEncoder = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
Projector = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


@dataclass(slots=True)
class LayerSource:
    raw_keys: torch.Tensor
    values: torch.Tensor
    rotated_keys: torch.Tensor
    hidden: torch.Tensor | None
    positions: torch.Tensor
    projector: Projector | None

    def gather(
        self,
        front: BlockFront,
        condition: Condition,
        position_encoder: PositionEncoder,
        reference: torch.Tensor,
        rotate_key: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if condition.pooling == "hidden":
            if self.hidden is None or self.projector is None:
                raise RuntimeError("hidden pooling requires captured hidden states and a projector")
            keys, values = self.projector(front.pool(self.hidden, 1))
        else:
            keys = front.pool(self.raw_keys, -2)
            values = front.pool(self.values, -2)
        if condition.position == "post_rope_pool":
            if condition.pooling != "kv":
                raise ValueError("post-RoPE pooling is only defined for KV pooling")
            return front.pool(self.rotated_keys, -2), values
        positions = front.pool(self.positions, -1)
        cos, sin = position_encoder(reference, positions)
        return rotate_key(keys, cos, sin), values


class E2Session:
    """State for one E2 condition; full KV cache is intentionally preserved."""

    def __init__(
        self,
        condition: Condition,
        *,
        seed: int = 42,
        area_ratios: tuple[float, float, float] = (0.50, 0.30, 0.20),
        trace_path: str | Path | None = None,
    ):
        self.condition = condition
        self.seed = seed
        self.area_ratios = area_ratios
        self.trace_path = Path(trace_path) if trace_path is not None else None
        self.position_encoder: PositionEncoder | None = None
        self.rotate_key: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None
        self.routed_layers: tuple[int, ...] = ()
        self.last_routed_layer = -1
        self.pending_sample_id = ""
        self.prefill_seconds = 0.0
        self.decode_seconds = 0.0
        self._forward_started: float | None = None
        self._forward_count = 0
        self._reset_prompt()

    @property
    def enabled(self) -> bool:
        return self.condition.pooled

    @property
    def configured(self) -> bool:
        return bool(self.visual_positions)

    def attach(
        self,
        layers: list[int],
        position_encoder: PositionEncoder,
        model_type: str,
    ) -> None:
        if not layers:
            raise ValueError("E2 found no full-attention layers")
        self.routed_layers = tuple(layers)
        self.last_routed_layer = layers[-1]
        self.position_encoder = position_encoder
        self.rotate_key = rotate_full_key if model_type == "qwen2_5_vl" else rotate_partial_key

    def _reset_prompt(self) -> None:
        self.sample_id = ""
        self.visual_positions: list[int] = []
        self.text_positions: list[int] = []
        self.post_visual_positions: list[int] = []
        self.grid: tuple[int, int] | None = None
        self.fine_positions: torch.Tensor | None = None
        self.current_positions: torch.Tensor | None = None
        self.sources: dict[int, LayerSource] = {}
        self.fixed_front: BlockFront | None = None
        self.step_front: BlockFront | None = None
        self.last_front: BlockFront | None = None
        self.decode_step = 0
        self.prompt_length = 0

    def begin_sample(self, sample_id: str) -> None:
        if not sample_id:
            raise ValueError("E2 sample_id must be non-empty")
        self.pending_sample_id = sample_id
        self.prefill_seconds = 0.0
        self.decode_seconds = 0.0
        self._forward_started = None
        self._forward_count = 0

    def start_forward(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._forward_started = time.perf_counter()

    def finish_forward(self) -> None:
        if self._forward_started is None:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._forward_started
        if self._forward_count == 0:
            self.prefill_seconds += elapsed
        else:
            self.decode_seconds += elapsed
        self._forward_count += 1
        self._forward_started = None

    def diagnostics(self) -> dict[str, Any]:
        front = self.fixed_front or self.step_front or self.last_front
        return {
            "sample_id": self.pending_sample_id,
            "grid": list(self.grid or ()),
            "active_tokens": front.node_count if front is not None else None,
            "compression_ratio": front.compression_ratio if front is not None else None,
            "front_hash": front.digest() if front is not None else None,
            "prefill_seconds": self.prefill_seconds,
            "decode_seconds": self.decode_seconds,
        }

    def configure_prompt(
        self,
        sample_id: str,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        image_token_id: int,
        spatial_merge_size: int,
    ) -> None:
        self._reset_prompt()
        self.pending_sample_id = sample_id
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("E2 requires batch size one")
        if image_grid_thw.shape != (1, 3) or int(image_grid_thw[0, 0]) != 1:
            raise ValueError("E2 requires exactly one image and does not support video")
        height = int(image_grid_thw[0, 1]) // spatial_merge_size
        width = int(image_grid_thw[0, 2]) // spatial_merge_size
        if height % 4 or width % 4:
            raise ValueError(f"E2 visual grid must be divisible by four, got {height}x{width}")
        tokens = input_ids[0].detach().cpu().tolist()
        self.visual_positions = [index for index, token in enumerate(tokens) if token == image_token_id]
        if len(self.visual_positions) != height * width:
            raise ValueError("processor visual token count does not match E2 grid")
        visual_set = set(self.visual_positions)
        self.text_positions = [index for index in range(len(tokens)) if index not in visual_set]
        last_visual = max(self.visual_positions)
        self.post_visual_positions = [index for index in self.text_positions if index > last_visual]
        if not self.post_visual_positions:
            raise ValueError("E2 prompt has no text query after the image")
        self.sample_id = sample_id
        self.grid = (height, width)
        self.prompt_length = input_ids.shape[-1]
        if self.condition.front_mode == "uniform":
            assert self.condition.block_size is not None
            self.fixed_front = BlockFront.uniform(height, width, self.condition.block_size)
        elif self.condition.front_mode == "random_fixed":
            self.fixed_front = self._random_front(None)
        self._trace("prompt", self.fixed_front)

    def observe_position_ids(self, position_ids: torch.Tensor | None) -> None:
        if position_ids is None or not self.configured:
            return
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        elif position_ids.shape[0] == 4:
            position_ids = position_ids[1:]
        self.current_positions = position_ids.detach()
        if position_ids.shape[-1] == self.prompt_length:
            index = torch.tensor(self.visual_positions, dtype=torch.long, device=position_ids.device)
            self.fine_positions = position_ids.index_select(-1, index).detach().float()

    def is_prefill_layer(self, layer: int) -> bool:
        return layer not in self.sources

    def capture_layer(
        self,
        layer: int,
        raw_keys: torch.Tensor,
        values: torch.Tensor,
        rotated_keys: torch.Tensor,
        hidden_states: torch.Tensor,
        projector: Projector | None,
    ) -> None:
        if self.fine_positions is None:
            raise RuntimeError("position IDs were not observed before E2 attention")
        index = torch.tensor(self.visual_positions, dtype=torch.long, device=raw_keys.device)
        self.sources[layer] = LayerSource(
            raw_keys.index_select(-2, index),
            values.index_select(-2, index),
            rotated_keys.index_select(-2, index),
            hidden_states.index_select(1, index) if self.condition.pooling == "hidden" else None,
            self.fine_positions.to(raw_keys.device),
            projector,
        )

    def prefill_compact(
        self, layer: int, full_keys: torch.Tensor, full_values: torch.Tensor, reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if self.fixed_front is None:
            return None
        return self._compact(layer, self.fixed_front, full_keys, full_values, reference, prefill=True)

    def decode_compact(
        self, layer: int, full_keys: torch.Tensor, full_values: torch.Tensor, reference: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        front = self.fixed_front
        if self.condition.front_mode == "random_perstep":
            if self.step_front is None:
                self.step_front = self._random_front(self.decode_step)
                self._trace("decode", self.step_front)
            front = self.step_front
        if front is None:
            raise RuntimeError("E2 decode front is unavailable")
        self.last_front = front
        keys, values, _, mask = self._compact(layer, front, full_keys, full_values, reference, prefill=False)
        return keys, values, mask

    def finish_layer(self, layer: int) -> None:
        if layer == self.last_routed_layer:
            self.decode_step += 1
            self.step_front = None

    def _compact(
        self,
        layer: int,
        front: BlockFront,
        full_keys: torch.Tensor,
        full_values: torch.Tensor,
        reference: torch.Tensor,
        *,
        prefill: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.position_encoder is None or self.rotate_key is None:
            raise RuntimeError("E2 session is not attached")
        visual_keys, visual_values = self.sources[layer].gather(
            front, self.condition, self.position_encoder, reference, self.rotate_key
        )
        text_index = torch.tensor(self.text_positions, dtype=torch.long, device=full_keys.device)
        if not prefill and full_keys.shape[-2] > self.prompt_length:
            generated = torch.arange(self.prompt_length, full_keys.shape[-2], device=full_keys.device)
            text_index = torch.cat((text_index, generated))
        text_keys = full_keys.index_select(-2, text_index)
        text_values = full_values.index_select(-2, text_index)
        keys = torch.cat((visual_keys, text_keys), dim=-2)
        values = torch.cat((visual_values, text_values), dim=-2)
        query_index = torch.tensor(
            self.post_visual_positions if prefill else [full_keys.shape[-2] - 1],
            dtype=torch.long,
            device=full_keys.device,
        )
        key_positions = torch.cat(
            (
                torch.full((front.node_count,), max(self.visual_positions), device=full_keys.device),
                text_index,
            )
        )
        mask = key_positions[None, :] <= query_index[:, None]
        return keys, values, query_index, mask[None, None]

    def _random_front(self, step: int | None) -> BlockFront:
        assert self.grid is not None
        return BlockFront.random_multiscale(
            *self.grid,
            stable_seed(self.seed, self.sample_id, step),
            self.area_ratios,
        )

    def _trace(self, phase: str, front: BlockFront | None) -> None:
        if self.trace_path is None or front is None:
            return
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        record: dict[str, Any] = {
            "sample_id": self.sample_id,
            "phase": phase,
            "decode_step": self.decode_step,
            "grid": list(self.grid or ()),
            "active_tokens": front.node_count,
            "compression_ratio": front.compression_ratio,
            "scale_counts": front.scale_counts,
            "front_hash": front.digest(),
            "nodes": [[node.y0, node.x0, node.size] for node in front.nodes],
        }
        with self.trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
