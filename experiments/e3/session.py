from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, cast

import torch

from experiments.e2.front import BlockFront
from experiments.e2.session import LayerSource, NativeLayerSource, PositionEncoder, Projector
from tokenfovea.integrations.qwen.common import (  # type: ignore[import-untyped]
    rotate_full_key,
    rotate_partial_key,
)

from .conditions import Condition


class E3Session:
    """Decode-only Text-Anchor state while preserving each pair's E2 prefill."""

    def __init__(self, condition: Condition, *, anchor_window: float = 2.0):
        if anchor_window < 0:
            raise ValueError("anchor_window must be non-negative")
        self.condition = condition
        self.anchor_window = anchor_window
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
        return self.condition.text_anchor or self.condition.compact

    @property
    def configured(self) -> bool:
        return bool(self.visual_positions) or self.native_capture_scale is not None

    @property
    def preserve_prefill(self) -> bool:
        return not self.condition.compact

    def attach(
        self,
        layers: list[int],
        position_encoder: PositionEncoder,
        model_type: str,
    ) -> None:
        if not layers:
            raise ValueError("E3 found no full-attention layers")
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
        self.sources: dict[int, LayerSource | NativeLayerSource] = {}
        self.front: BlockFront | None = None
        self.prompt_length = 0
        self.decode_step = 0
        self.native_capture_scale: int | None = None
        self.native_capture_positions: list[int] = []
        self.native_capture_grid: tuple[int, int] | None = None
        self.native_layers: dict[
            int,
            dict[int, tuple[torch.Tensor, torch.Tensor, tuple[int, int]]],
        ] = {}
        self.native_preparing = False
        self.native_prefill_seconds = 0.0
        self.native_bank_tokens = 0
        self.anchor_min: float | None = None
        self.anchor_max: float | None = None

    def begin_sample(self, sample_id: str) -> None:
        if not sample_id:
            raise ValueError("E3 sample_id must be non-empty")
        self._reset_prompt()
        self.native_preparing = self.condition.native
        self.pending_sample_id = sample_id
        self.prefill_seconds = 0.0
        self.decode_seconds = 0.0
        self._forward_started = None
        self._forward_count = 0

    def begin_native_capture(self, area_scale: int) -> None:
        if not self.condition.native or not self.native_preparing:
            raise RuntimeError("native capture requires a prepared E3 Native2 sample")
        if area_scale != 4:
            raise ValueError("E3 Native2 captures only area scale 4")
        if self.native_capture_scale is not None:
            raise RuntimeError("another E3 native capture is active")
        self.native_capture_scale = area_scale
        self.native_capture_positions = []
        self.native_capture_grid = None
        self.native_layers[area_scale] = {}

    def end_native_capture(self) -> None:
        if self.native_capture_scale is None:
            raise RuntimeError("no E3 native capture is active")
        if set(self.native_layers[self.native_capture_scale]) != set(self.routed_layers):
            raise RuntimeError("E3 native prefill did not capture every routed layer")
        self.native_capture_scale = None
        self.native_capture_positions = []
        self.native_capture_grid = None

    def abort_native_sample(self) -> None:
        self._reset_prompt()

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
        if self.native_capture_scale is not None:
            self.native_prefill_seconds += elapsed
        elif self._forward_count == 0:
            self.prefill_seconds += elapsed
            self._forward_count += 1
        else:
            self.decode_seconds += elapsed
            self._forward_count += 1
        self._forward_started = None

    def diagnostics(self) -> dict[str, Any]:
        active = self.front.node_count if self.front is not None else None
        return {
            "sample_id": self.pending_sample_id,
            "grid": list(self.grid or ()),
            "active_tokens": active,
            "compression_ratio": (
                active / (self.grid[0] * self.grid[1])
                if active is not None and self.grid is not None
                else None
            ),
            "prefill_seconds": self.prefill_seconds,
            "decode_seconds": self.decode_seconds,
            "native_prefill_seconds": self.native_prefill_seconds,
            "native_bank_tokens": self.native_bank_tokens or None,
            "anchor_position_min": self.anchor_min,
            "anchor_position_max": self.anchor_max,
            "decode_steps": self.decode_step,
        }

    def configure_prompt(
        self,
        sample_id: str,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        image_token_id: int,
        spatial_merge_size: int,
    ) -> None:
        native_layers = self.native_layers if self.native_preparing else {}
        native_seconds = self.native_prefill_seconds
        self._reset_prompt()
        self.native_layers = native_layers
        self.native_prefill_seconds = native_seconds
        self.pending_sample_id = sample_id
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("E3 requires batch size one")
        if image_grid_thw.shape != (1, 3) or int(image_grid_thw[0, 0]) != 1:
            raise ValueError("E3 requires exactly one image and does not support video")
        height = int(image_grid_thw[0, 1]) // spatial_merge_size
        width = int(image_grid_thw[0, 2]) // spatial_merge_size
        if height <= 0 or width <= 0:
            raise ValueError("E3 visual grid must be positive")
        if self.condition.compact and (height % 2 or width % 2):
            raise ValueError("E3 Pool2/Native2 grids must be divisible by two")
        tokens = input_ids[0].detach().cpu().tolist()
        self.visual_positions = [index for index, token in enumerate(tokens) if token == image_token_id]
        if len(self.visual_positions) != height * width:
            raise ValueError("processor visual token count does not match E3 grid")
        visual_set = set(self.visual_positions)
        self.text_positions = [index for index in range(len(tokens)) if index not in visual_set]
        last_visual = max(self.visual_positions)
        self.post_visual_positions = [index for index in self.text_positions if index > last_visual]
        if self.condition.compact and not self.post_visual_positions:
            raise ValueError("E3 compact prefill has no text query after the image")
        self.sample_id = sample_id
        self.grid = (height, width)
        self.front = BlockFront.uniform(height, width, self.condition.block_size)
        self.prompt_length = input_ids.shape[-1]
        if self.condition.native:
            if set(self.native_layers) != {4}:
                raise RuntimeError("E3 Native2 requires the prepared scale-4 bank")
            captured_grids = {source[2] for source in self.native_layers[4].values()}
            if captured_grids != {(height // 2, width // 2)}:
                raise ValueError(
                    "E3 Native2 auxiliary grid must be exactly half the main grid"
                )
            self.native_preparing = False
            self.native_bank_tokens = height * width + height * width // 4

    def configure_native_capture_prompt(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        image_token_id: int,
        spatial_merge_size: int,
    ) -> None:
        if self.native_capture_scale != 4:
            raise RuntimeError("E3 Native2 capture is not active")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("E3 requires batch size one")
        if image_grid_thw.shape != (1, 3) or int(image_grid_thw[0, 0]) != 1:
            raise ValueError("E3 native capture requires exactly one image")
        height = int(image_grid_thw[0, 1]) // spatial_merge_size
        width = int(image_grid_thw[0, 2]) // spatial_merge_size
        positions = [
            index
            for index, token in enumerate(input_ids[0].detach().cpu().tolist())
            if token == image_token_id
        ]
        if len(positions) != height * width:
            raise ValueError("E3 native visual token count does not match its grid")
        self.native_capture_positions = positions
        self.native_capture_grid = (height, width)

    def observe_position_ids(self, position_ids: torch.Tensor | None) -> None:
        if position_ids is None or not self.configured:
            return
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        elif position_ids.shape[0] == 4:
            position_ids = position_ids[1:]
        self.current_positions = position_ids.detach()
        if position_ids.shape[-1] == self.prompt_length and self.visual_positions:
            index = torch.tensor(self.visual_positions, dtype=torch.long, device=position_ids.device)
            self.fine_positions = position_ids.index_select(-1, index).detach().float()

    def is_prefill_layer(self, layer: int) -> bool:
        if self.native_capture_scale is not None:
            return layer not in self.native_layers[self.native_capture_scale]
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
        del hidden_states, projector
        if self.native_capture_scale is not None:
            if self.native_capture_grid is None:
                raise RuntimeError("E3 native capture prompt was not configured")
            index = torch.tensor(
                self.native_capture_positions, dtype=torch.long, device=raw_keys.device
            )
            self.native_layers[4][layer] = (
                raw_keys.index_select(-2, index).detach(),
                values.index_select(-2, index).detach(),
                self.native_capture_grid,
            )
            return
        if self.fine_positions is None or self.grid is None:
            raise RuntimeError("E3 position IDs were not observed before attention")
        index = torch.tensor(self.visual_positions, dtype=torch.long, device=raw_keys.device)
        if self.condition.native:
            native_keys, native_values, native_grid = self.native_layers[4].pop(layer)
            self.sources[layer] = NativeLayerSource(
                {
                    1: raw_keys.index_select(-2, index).detach(),
                    4: native_keys,
                },
                {
                    1: values.index_select(-2, index).detach(),
                    4: native_values,
                },
                {1: self.grid, 4: native_grid},
                self.fine_positions.to(raw_keys.device),
            )
        else:
            self.sources[layer] = LayerSource(
                raw_keys.index_select(-2, index).detach(),
                values.index_select(-2, index).detach(),
                rotated_keys.index_select(-2, index).detach(),
                None,
                self.fine_positions.to(raw_keys.device),
                None,
            )

    def prefill_compact(
        self,
        layer: int,
        full_keys: torch.Tensor,
        full_values: torch.Tensor,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if self.native_capture_scale is not None:
            return None
        if self.preserve_prefill:
            return None
        if self.front is None:
            raise RuntimeError("E3 prefill front is unavailable")
        return self._compact_center(layer, self.front, full_keys, full_values, reference)

    def decode_compact(
        self,
        layer: int,
        full_keys: torch.Tensor,
        full_values: torch.Tensor,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.front is None:
            raise RuntimeError("E3 decode front is unavailable")
        visual_keys, visual_values = (
            self._anchor_visual(layer, self.front, reference)
            if self.condition.text_anchor
            else self._center_visual(layer, self.front, reference)
        )
        if not self.condition.compact:
            index = torch.tensor(self.visual_positions, dtype=torch.long, device=full_keys.device)
            keys = full_keys.clone()
            keys.index_copy_(-2, index, visual_keys)
            mask = torch.ones(
                (1, 1, 1, full_keys.shape[-2]),
                dtype=torch.bool,
                device=full_keys.device,
            )
            return keys, full_values, mask
        text_index = self._text_index(full_keys.shape[-2], full_keys.device)
        keys = torch.cat((visual_keys, full_keys.index_select(-2, text_index)), dim=-2)
        values = torch.cat((visual_values, full_values.index_select(-2, text_index)), dim=-2)
        query_position = full_keys.shape[-2] - 1
        key_positions = torch.cat(
            (
                torch.full((self.front.node_count,), max(self.visual_positions), device=full_keys.device),
                text_index,
            )
        )
        mask = (key_positions[None, :] <= query_position)[None, None]
        return keys, values, mask

    def finish_layer(self, layer: int) -> None:
        if layer == self.last_routed_layer:
            self.decode_step += 1

    def _compact_center(
        self,
        layer: int,
        front: BlockFront,
        full_keys: torch.Tensor,
        full_values: torch.Tensor,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.position_encoder is None or self.rotate_key is None:
            raise RuntimeError("E3 session is not attached")
        visual_keys, visual_values = self.sources[layer].gather(
            front,
            cast(Any, self.condition),
            self.position_encoder,
            reference,
            self.rotate_key,
        )
        text_index = torch.tensor(self.text_positions, dtype=torch.long, device=full_keys.device)
        keys = torch.cat((visual_keys, full_keys.index_select(-2, text_index)), dim=-2)
        values = torch.cat((visual_values, full_values.index_select(-2, text_index)), dim=-2)
        query_index = torch.tensor(
            self.post_visual_positions, dtype=torch.long, device=full_keys.device
        )
        key_positions = torch.cat(
            (
                torch.full((front.node_count,), max(self.visual_positions), device=full_keys.device),
                text_index,
            )
        )
        mask = key_positions[None, :] <= query_index[:, None]
        return keys, values, query_index, mask[None, None]

    def _anchor_visual(
        self,
        layer: int,
        front: BlockFront,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.position_encoder is None or self.rotate_key is None or self.current_positions is None:
            raise RuntimeError("E3 Text-Anchor position state is unavailable")
        source = self.sources[layer]
        if isinstance(source, NativeLayerSource):
            raw_keys, values = self._native_raw(source, front)
        else:
            raw_keys = front.pool(source.raw_keys, -2)
            values = front.pool(source.values, -2)
        positions = front.pool(source.positions.to(raw_keys.device), -1)
        anchors = self._anchor_positions(positions, raw_keys.device)
        cos, sin = self.position_encoder(reference, anchors)
        return self.rotate_key(raw_keys, cos, sin), values

    def _center_visual(
        self,
        layer: int,
        front: BlockFront,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.position_encoder is None or self.rotate_key is None:
            raise RuntimeError("E3 center-position state is unavailable")
        return self.sources[layer].gather(
            front,
            cast(Any, self.condition),
            self.position_encoder,
            reference,
            self.rotate_key,
        )

    @staticmethod
    def _native_raw(
        source: NativeLayerSource,
        front: BlockFront,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if any(node.size != 2 for node in front.nodes):
            raise ValueError("E3 Native2 front must contain only 2x2 nodes")
        raw = source.raw_keys[1]
        keys = raw.new_empty((*raw.shape[:-2], front.node_count, raw.shape[-1]))
        base_values = source.values[1]
        values = base_values.new_empty(
            (*base_values.shape[:-2], front.node_count, base_values.shape[-1])
        )
        output = torch.arange(front.node_count, dtype=torch.long, device=raw.device)
        source_grid = source.grids[4]
        indices = torch.tensor(
            [
                (node.y0 // 2) * source_grid[1] + node.x0 // 2
                for node in front.nodes
            ],
            dtype=torch.long,
            device=raw.device,
        )
        keys.index_copy_(-2, output, source.raw_keys[4].index_select(-2, indices))
        values.index_copy_(-2, output, source.values[4].index_select(-2, indices))
        return keys, values

    def _anchor_positions(self, positions: torch.Tensor, device: torch.device) -> torch.Tensor:
        assert self.current_positions is not None
        current = self.current_positions[..., -1:].to(device=device, dtype=torch.float32)
        count = positions.shape[-1]
        anchors = current.expand(-1, -1, count).clone()
        spatial = positions[1:].to(device=device, dtype=torch.float32)
        minimum = spatial.amin(dim=-1, keepdim=True)
        span = spatial.amax(dim=-1, keepdim=True) - minimum
        normalized = torch.where(
            span > 0,
            (spatial - minimum) / span,
            torch.full_like(spatial, 0.5),
        )
        window = self.anchor_window
        anchors[1:] = current[1:] - window + (window + 1.0) * (
            1.0 + (count - 1) * normalized
        ) / (count + 1)
        anchored_spatial = anchors[1:]
        minimum, maximum = float(anchored_spatial.min()), float(anchored_spatial.max())
        self.anchor_min = minimum if self.anchor_min is None else min(self.anchor_min, minimum)
        self.anchor_max = maximum if self.anchor_max is None else max(self.anchor_max, maximum)
        return anchors

    def _text_index(self, full_length: int, device: torch.device) -> torch.Tensor:
        prompt = torch.tensor(self.text_positions, dtype=torch.long, device=device)
        generated = torch.arange(self.prompt_length, full_length, dtype=torch.long, device=device)
        return torch.cat((prompt, generated))
