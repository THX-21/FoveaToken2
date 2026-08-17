from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import torch

from .config import FoveaConfig
from .pyramid import LayerKVPyramid, PositionEncoder, RotateKey
from .router import SplitMergeRouter
from .topology import DeviceTreeTopology, VisualTokenForest


class FoveaSession:
    """Routing state for one prompt shared by patched decoder layers."""

    def __init__(self, config: FoveaConfig):
        self.config = config
        self.selected_heads = self._load_signal_selection(config.signal_selection)
        self._head_index_cache: dict[tuple[int, torch.device], torch.Tensor] = {}
        self.position_encoder: PositionEncoder | None = None
        self.rotate_key: RotateKey | None = None
        self.routed_layers: tuple[int, ...] = ()
        self.last_routed_layer = -1
        self._reset_state()

    @staticmethod
    def _load_signal_selection(path: str | None) -> dict[int, tuple[int, ...]] | None:
        if path is None:
            return None
        selection_path = Path(path)
        if not selection_path.exists():
            raise FileNotFoundError(f"signal selection not found: {selection_path}")
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
        pairs = payload.get("selected_heads")
        if not isinstance(pairs, list) or not pairs:
            raise ValueError("signal selection must contain a non-empty selected_heads list")
        selected: dict[int, set[int]] = {}
        for pair in pairs:
            layer, head = int(pair["layer"]), int(pair["head"])
            if layer < 0 or head < 0:
                raise ValueError("selected layer/head indices must be non-negative")
            selected.setdefault(layer, set()).add(head)
        return {layer: tuple(sorted(heads)) for layer, heads in selected.items()}

    def _reset_state(self) -> None:
        self.forest: VisualTokenForest | None = None
        self.topology: DeviceTreeTopology | None = None
        self._topologies: dict[torch.device, DeviceTreeTopology] = {}
        self._routing_device: torch.device | None = None
        self.router: SplitMergeRouter | None = None
        self.pyramids: dict[int, LayerKVPyramid] = {}
        self.visual_positions: list[int] = []
        self.prompt_text_positions: list[int] = []
        self.fine_position_ids: torch.Tensor | None = None
        self.current_position_ids: torch.Tensor | None = None
        self.prompt_length = 0
        self.prompt_signal_sum: torch.Tensor | None = None
        self.prompt_signal_count = 0
        self.step_signal_sum: torch.Tensor | None = None
        self.step_signal_count = 0
        self.ema_leaf_scores: torch.Tensor | None = None
        self.decode_step = 0
        self._visual_indices: dict[torch.device, torch.Tensor] = {}
        self._prompt_text_indices: dict[torch.device, torch.Tensor] = {}
        self._decode_text_indices: dict[tuple[torch.device, int], torch.Tensor] = {}
        self._decode_active_ids: torch.Tensor | None = None
        self._decode_anchor_positions: dict[torch.device, torch.Tensor] = {}
        self.native_capture_scale: int | None = None
        self.native_capture_visual_positions: list[int] = []
        self.native_capture_grids: tuple[tuple[int, int], ...] = ()
        self.native_sources: dict[
            int,
            dict[int, tuple[torch.Tensor, torch.Tensor, tuple[tuple[int, int], ...]]],
        ] = {}
        self.native_preparing = False

    def reset_prompt(self) -> None:
        """Discard all state owned by the previous prompt."""
        self._reset_state()

    def begin_native_sample(self) -> None:
        """Reset the prompt and accept auxiliary native-resolution prefills."""
        if self.config.pooling_mode != "native_multiscale":
            raise RuntimeError("native sample preparation requires native_multiscale pooling")
        self._reset_state()
        self.native_preparing = True

    def begin_native_capture(self, area_scale: int) -> None:
        if not self.native_preparing:
            raise RuntimeError("call begin_native_sample() before auxiliary native prefills")
        if area_scale not in {4, 16, 64}:
            raise ValueError("auxiliary native scale must be 4, 16, or 64")
        if self.native_capture_scale is not None:
            raise RuntimeError("a native scale capture is already active")
        self.native_capture_scale = area_scale
        self.native_capture_visual_positions = []
        self.native_capture_grids = ()
        self.native_sources[area_scale] = {}

    def end_native_capture(self) -> None:
        if self.native_capture_scale is None:
            raise RuntimeError("no native scale capture is active")
        if set(self.native_sources[self.native_capture_scale]) != set(self.routed_layers):
            raise RuntimeError("native auxiliary prefill did not capture every routed layer")
        self.native_capture_scale = None
        self.native_capture_visual_positions = []
        self.native_capture_grids = ()

    def abort_native_sample(self) -> None:
        self._reset_state()

    def attach(
        self,
        routed_layers: list[int],
        position_encoder: PositionEncoder,
        rotate_key: RotateKey,
    ) -> None:
        if not routed_layers:
            raise ValueError("no full-attention layers were found")
        self.routed_layers = tuple(routed_layers)
        self.last_routed_layer = routed_layers[-1]
        if self.selected_heads is not None:
            missing = sorted(set(self.selected_heads) - set(routed_layers))
            if missing:
                raise ValueError(f"selected signal layers are not routed full-attention layers: {missing}")
        self.position_encoder = position_encoder
        self.rotate_key = rotate_key

    @property
    def is_configured(self) -> bool:
        return self.forest is not None or self.native_capture_scale is not None

    @property
    def enabled(self) -> bool:
        return self.config.mode != "full"

    @property
    def dynamic(self) -> bool:
        return self.config.mode == "dynamic"

    def needs_signal(self, layer_idx: int) -> bool:
        return (
            self.native_capture_scale is None
            and self.dynamic
            and (self.selected_heads is None or layer_idx in self.selected_heads)
        )

    def is_prefill_layer(self, layer_idx: int) -> bool:
        if self.native_capture_scale is not None:
            return layer_idx not in self.native_sources[self.native_capture_scale]
        return layer_idx not in self.pyramids

    @staticmethod
    def validate_decode_batch(batch_size: int) -> None:
        if batch_size != 1:
            raise ValueError("TokenFovea does not support beam search or decode batch expansion")

    def observe_position_ids(self, position_ids: torch.Tensor | None) -> None:
        if position_ids is None or self.forest is None:
            return
        if position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        elif position_ids.shape[0] == 4:
            position_ids = position_ids[1:]
        self.current_position_ids = position_ids.detach()
        if position_ids.shape[-1] == self.prompt_length:
            visual_index = self.visual_index(position_ids.device)
            self.fine_position_ids = position_ids.index_select(-1, visual_index).detach()

    def configure_prompt(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        image_token_id: int,
        spatial_merge_size: int,
    ) -> None:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("TokenFovea requires batch_size=1")
        native_sources = self.native_sources if self.native_preparing else {}
        self._reset_state()
        if native_sources:
            self.native_sources = native_sources
        token_ids = input_ids[0].detach().cpu().tolist()
        self.visual_positions = [i for i, token_id in enumerate(token_ids) if token_id == image_token_id]
        visual_set = set(self.visual_positions)
        self.prompt_text_positions = [i for i in range(len(token_ids)) if i not in visual_set]
        grids = []
        for temporal, height, width in image_grid_thw.detach().cpu().tolist():
            if int(temporal) != 1:
                raise ValueError("TokenFovea currently supports images only")
            if int(height) % spatial_merge_size or int(width) % spatial_merge_size:
                raise ValueError("vision grid is not divisible by spatial_merge_size")
            grids.append((int(height) // spatial_merge_size, int(width) // spatial_merge_size))
        expected = sum(height * width for height, width in grids)
        if expected != len(self.visual_positions):
            raise ValueError(
                f"processor produced {len(self.visual_positions)} image tokens but grid describes {expected}"
            )
        self.forest = (
            VisualTokenForest.from_aligned_grids(grids, max_block_size=8)
            if self.config.pooling_mode == "native_multiscale"
            else VisualTokenForest.from_grids(grids)
        )
        self.prompt_length = input_ids.shape[1]
        if self.config.pooling_mode == "native_multiscale":
            if set(self.native_sources) != {4, 16, 64}:
                raise RuntimeError("native_multiscale requires prepared scale 4/16/64 banks")
            self.native_preparing = False

    def configure_native_capture_prompt(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        image_token_id: int,
        spatial_merge_size: int,
    ) -> None:
        if self.native_capture_scale is None:
            raise RuntimeError("native capture scale is not active")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("TokenFovea requires batch_size=1")
        grids = []
        for temporal, height, width in image_grid_thw.detach().cpu().tolist():
            if int(temporal) != 1:
                raise ValueError("TokenFovea native multiscale supports images only")
            if int(height) % spatial_merge_size or int(width) % spatial_merge_size:
                raise ValueError("vision grid is not divisible by spatial_merge_size")
            grids.append((int(height) // spatial_merge_size, int(width) // spatial_merge_size))
        tokens = input_ids[0].detach().cpu().tolist()
        positions = [index for index, token in enumerate(tokens) if token == image_token_id]
        if len(positions) != sum(h * w for h, w in grids):
            raise ValueError("native auxiliary visual token count does not match its grids")
        self.native_capture_visual_positions = positions
        self.native_capture_grids = tuple(grids)

    @staticmethod
    def _index(values: list[int], device: torch.device) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.long, device=device)

    def visual_index(self, device: torch.device) -> torch.Tensor:
        if self.native_capture_scale is not None:
            return self._index(self.native_capture_visual_positions, device)
        device = torch.device(device)
        index = self._visual_indices.get(device)
        if index is None:
            index = self._index(self.visual_positions, device)
            self._visual_indices[device] = index
        return index

    def _ensure_device_state(self, device: torch.device) -> DeviceTreeTopology:
        device = torch.device(device)
        topology = self._topologies.get(device)
        if topology is None:
            assert self.forest is not None
            topology = DeviceTreeTopology.build(self.forest, device)
            self._topologies[device] = topology
            self._prompt_text_indices[device] = self._index(self.prompt_text_positions, device)
        if self.router is not None:
            return topology
        assert self.forest is not None
        self.topology = topology
        self._routing_device = device
        initial = self._index(sorted(self.forest.initial_front(self.config.budget)), device)
        self.router = SplitMergeRouter(
            topology,
            initial,
            epsilon=self.config.epsilon,
            max_swaps=self.config.max_swaps,
            score_mode=self.config.score_mode,
        )
        return topology

    def _accumulate_signal(
        self,
        current: torch.Tensor | None,
        signal: torch.Tensor,
    ) -> torch.Tensor:
        assert self._routing_device is not None
        signal = signal.to(self._routing_device)
        if current is None:
            return signal
        if self.config.signal_aggregation == "max":
            return torch.maximum(current, signal)
        return current + signal

    def capture_prefill_layer(
        self,
        layer_idx: int,
        raw_keys: torch.Tensor,
        values: torch.Tensor,
        rotated_keys: torch.Tensor,
        visual_attention: torch.Tensor | None,
        hidden_states: torch.Tensor | None = None,
        projector: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> None:
        if self.native_capture_scale is not None:
            visual_index = self.visual_index(raw_keys.device)
            self.native_sources[self.native_capture_scale][layer_idx] = (
                raw_keys.index_select(-2, visual_index).detach(),
                values.index_select(-2, visual_index).detach(),
                self.native_capture_grids,
            )
            return
        if self.forest is None or self.fine_position_ids is None:
            return
        topology = self._ensure_device_state(raw_keys.device)
        visual_index = self.visual_index(raw_keys.device)
        fine_positions = self.fine_position_ids.to(device=raw_keys.device, dtype=torch.float32)
        if self.config.pooling_mode == "native_multiscale":
            fine_raw = raw_keys.index_select(-2, visual_index).detach()
            fine_values = values.index_select(-2, visual_index).detach()
            scale_sources = {
                1: (fine_raw, fine_values, self.forest.grids),
                **{
                    scale: self.native_sources[scale][layer_idx]
                    for scale in (4, 16, 64)
                },
            }
            self.pyramids[layer_idx] = LayerKVPyramid.from_native_scales(
                topology,
                self.forest,
                scale_sources,
                fine_positions,
            )
            for scale in (4, 16, 64):
                del self.native_sources[scale][layer_idx]
        elif self.config.pooling_mode == "hidden":
            if hidden_states is None or projector is None:
                raise RuntimeError("hidden pooling requires hidden states and a KV projector")
            fine_hidden = hidden_states.index_select(1, visual_index)
            self.pyramids[layer_idx] = LayerKVPyramid.from_hidden(
                topology,
                fine_hidden,
                fine_positions,
                projector,
            )
        else:
            fine_raw = raw_keys.index_select(-2, visual_index)
            fine_values = values.index_select(-2, visual_index)
            fine_rotated = (
                rotated_keys.index_select(-2, visual_index)
                if self.config.position_mode == "post_rope_pool"
                else None
            )
            self.pyramids[layer_idx] = LayerKVPyramid.from_kv(
                topology,
                fine_raw,
                fine_values,
                fine_positions,
                fine_rotated,
            )
        if visual_attention is not None:
            reduced = self._reduce_signal(layer_idx, visual_attention)
            if reduced is not None:
                signal, count = reduced
                self.prompt_signal_sum = self._accumulate_signal(self.prompt_signal_sum, signal)
                self.prompt_signal_count += count
        if layer_idx == self.last_routed_layer and self.dynamic:
            self._finish_prefill()

    def _finish_prefill(self) -> None:
        if (
            self.config.mode != "dynamic"
            or not self.config.route_after_prefill
            or self.prompt_signal_sum is None
            or self.router is None
            or self.topology is None
        ):
            return
        leaf_scores = self.prompt_signal_sum
        if self.config.signal_aggregation == "mean":
            leaf_scores = leaf_scores / self.prompt_signal_count
        leaf_scores = leaf_scores / leaf_scores.sum().clamp_min(1e-12)
        scores = self.topology.aggregate_leaves(leaf_scores, density=self.config.score_mode == "density")
        self.router.step(scores)

    def _text_index(self, full_length: int, device: torch.device) -> torch.Tensor:
        device = torch.device(device)
        cache_key = (device, full_length)
        index = self._decode_text_indices.get(cache_key)
        if index is None:
            prompt_text_index = self._prompt_text_indices.get(device)
            if prompt_text_index is None:
                prompt_text_index = self._index(self.prompt_text_positions, device)
                self._prompt_text_indices[device] = prompt_text_index
            generated = torch.arange(self.prompt_length, full_length, dtype=torch.long, device=device)
            index = torch.cat((prompt_text_index, generated))
            self._decode_text_indices[cache_key] = index
        return index

    def _anchor_positions(self, active_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        device = torch.device(device)
        cached = self._decode_anchor_positions.get(device)
        if cached is not None:
            return cached
        if self.current_position_ids is None:
            raise RuntimeError("text_anchor requires current position ids")
        current = self.current_position_ids[..., -1:].to(device=device, dtype=torch.float32)
        pyramid = next(iter(self.pyramids.values()))
        selected_positions = pyramid.native_positions.to(device=device).index_select(-1, active_ids)
        spatial = selected_positions[1:]
        positions = current.expand(-1, -1, active_ids.numel()).clone()
        window = self.config.anchor_window
        minimum = spatial.amin(dim=-1, keepdim=True)
        span = spatial.amax(dim=-1, keepdim=True) - minimum
        normalized = torch.where(
            span > 0,
            (spatial - minimum) / span,
            torch.full_like(spatial, 0.5),
        )
        count = active_ids.numel()
        positions[1:] = current[1:] - window + (window + 1.0) * (
            1.0 + (count - 1) * normalized
        ) / (count + 1)
        self._decode_anchor_positions[device] = positions
        return positions

    def compose(
        self,
        layer_idx: int,
        full_keys: torch.Tensor,
        full_values: torch.Tensor,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            self.router is None
            or layer_idx not in self.pyramids
            or self.position_encoder is None
            or self.rotate_key is None
        ):
            return full_keys, full_values, full_keys.new_empty((0,), dtype=torch.long)
        if self._decode_active_ids is None:
            self._decode_active_ids = self.router.active_ids()
        active_ids = self._decode_active_ids.to(full_keys.device)
        anchors = (
            self._anchor_positions(active_ids, full_keys.device)
            if self.config.position_mode == "text_anchor"
            else None
        )
        visual_keys, visual_values = self.pyramids[layer_idx].gather(
            active_ids,
            self.config.position_mode,
            self.rotate_key,
            self.position_encoder,
            reference,
            anchors,
        )
        text_index = self._text_index(full_keys.shape[-2], full_keys.device)
        text_keys = full_keys.index_select(-2, text_index)
        text_values = full_values.index_select(-2, text_index)
        return (
            torch.cat((visual_keys, text_keys), dim=-2),
            torch.cat((visual_values, text_values), dim=-2),
            active_ids,
        )

    def compose_attention_mask(
        self,
        attention_mask: torch.Tensor | None,
        active_ids: torch.Tensor,
        full_length: int,
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        if attention_mask.ndim < 2 or attention_mask.shape[-1] < full_length:
            raise ValueError("attention mask does not cover the full KV cache")
        text_index = self._text_index(full_length, attention_mask.device)
        text_mask = attention_mask.index_select(-1, text_index)
        visual_shape = list(text_mask.shape)
        visual_shape[-1] = active_ids.numel()
        if attention_mask.dtype == torch.bool or not attention_mask.is_floating_point():
            visual_mask = torch.ones(visual_shape, dtype=attention_mask.dtype, device=attention_mask.device)
        elif attention_mask.ndim == 2:
            visual_mask = torch.ones(visual_shape, dtype=attention_mask.dtype, device=attention_mask.device)
        else:
            visual_mask = torch.zeros(visual_shape, dtype=attention_mask.dtype, device=attention_mask.device)
        return torch.cat((visual_mask, text_mask), dim=-1)

    def record_decode_layer(self, layer_idx: int, visual_attention: torch.Tensor | None) -> None:
        if visual_attention is not None:
            reduced = self._reduce_signal(layer_idx, visual_attention)
            if reduced is not None:
                signal, count = reduced
                self.step_signal_sum = self._accumulate_signal(self.step_signal_sum, signal)
                self.step_signal_count += count
        if layer_idx == self.last_routed_layer:
            self._finish_decode_step()

    def _reduce_signal(self, layer_idx: int, signal: torch.Tensor) -> tuple[torch.Tensor, int] | None:
        signal = signal.detach().float()
        if signal.ndim == 1:
            signal = signal.unsqueeze(0)
        if self.selected_heads is None:
            selected = signal
        else:
            heads = self.selected_heads.get(layer_idx)
            if not heads:
                return None
            if heads[-1] >= signal.shape[0]:
                raise ValueError(f"selected head {heads[-1]} is out of range for layer {layer_idx}")
            cache_key = (layer_idx, signal.device)
            index = self._head_index_cache.get(cache_key)
            if index is None:
                index = torch.tensor(heads, dtype=torch.long, device=signal.device)
                self._head_index_cache[cache_key] = index
            selected = signal.index_select(0, index)
        if self.config.signal_aggregation == "max":
            return selected.max(dim=0).values, 1
        return selected.sum(dim=0), selected.shape[0]

    def _finish_decode_step(self) -> None:
        self.decode_step += 1
        if (
            self.dynamic
            and self.step_signal_sum is not None
            and self._decode_active_ids is not None
            and self.router is not None
            and self.topology is not None
        ):
            active_scores = self.step_signal_sum
            if self.config.signal_aggregation == "mean":
                active_scores = active_scores / self.step_signal_count
            active_scores = active_scores / active_scores.sum().clamp_min(1e-12)
            node_scores, leaf_scores = self.router.scores_from_active(self._decode_active_ids, active_scores)
            if self.config.attention_ema:
                if self.ema_leaf_scores is not None:
                    alpha = self.config.attention_ema
                    leaf_scores = alpha * self.ema_leaf_scores + (1.0 - alpha) * leaf_scores
                    node_scores = self.topology.aggregate_leaves(
                        leaf_scores,
                        density=self.config.score_mode == "density",
                    )
                self.ema_leaf_scores = leaf_scores
            if self.decode_step % self.config.update_interval == 0:
                self.router.step(node_scores)
        self.step_signal_sum = None
        self.step_signal_count = 0
        self._decode_active_ids = None
        self._decode_anchor_positions.clear()
        self._decode_text_indices.clear()
