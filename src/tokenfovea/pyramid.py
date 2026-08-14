from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .topology import DeviceTreeTopology, VisualTokenForest

RotateKey = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
PositionEncoder = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


def _pool_nodes(fine: torch.Tensor, topology: DeviceTreeTopology, node_dim: int) -> torch.Tensor:
    """Pool a complete tree a level at a time instead of launching one kernel per node."""
    node_dim %= fine.ndim
    shape = list(fine.shape)
    shape[node_dim] = topology.node_count
    nodes = fine.new_empty(shape)
    nodes.index_copy_(node_dim, topology.leaf_nodes, fine)
    for parent_ids, child_ids, weights in topology.levels:
        flat_children = child_ids.reshape(-1)
        gathered = nodes.index_select(node_dim, flat_children)
        shape = list(gathered.shape)
        shape[node_dim] = child_ids.shape[0]
        shape.insert(node_dim + 1, child_ids.shape[1])
        gathered = gathered.reshape(shape)
        weight_shape = [1] * gathered.ndim
        weight_shape[node_dim] = child_ids.shape[0]
        weight_shape[node_dim + 1] = child_ids.shape[1]
        pooled = (gathered * weights.to(fine.dtype).view(weight_shape)).sum(dim=node_dim + 1)
        nodes.index_copy_(node_dim, parent_ids, pooled)
    return nodes


@dataclass(slots=True)
class LayerKVPyramid:
    """One decoder layer's pre-RoPE multi-scale visual KV pyramid."""

    raw_keys: torch.Tensor
    values: torch.Tensor
    native_positions: torch.Tensor
    post_rope_keys: torch.Tensor | None = None

    @classmethod
    def from_native_scales(
        cls,
        topology: DeviceTreeTopology,
        forest: VisualTokenForest,
        scale_sources: dict[int, tuple[torch.Tensor, torch.Tensor, tuple[tuple[int, int], ...]]],
        fine_positions: torch.Tensor,
    ) -> LayerKVPyramid:
        """Assemble tree-aligned nodes from native 1/4/16/64-resolution K/V."""
        required = {1, 4, 16, 64}
        if set(scale_sources) != required:
            raise ValueError(f"native scale sources must be exactly {sorted(required)}")
        first_keys, first_values, _ = scale_sources[1]
        if first_keys.ndim != 4 or first_values.shape != first_keys.shape:
            raise ValueError("native K/V must have shape [batch, kv_heads, tokens, head_dim]")
        raw_keys = first_keys.new_empty((*first_keys.shape[:-2], topology.node_count, first_keys.shape[-1]))
        values = first_values.new_empty((*first_values.shape[:-2], topology.node_count, first_values.shape[-1]))

        node_ids_by_scale: dict[int, list[int]] = {scale: [] for scale in required}
        source_indices_by_scale: dict[int, list[int]] = {scale: [] for scale in required}
        for node in forest.nodes:
            height, width = node.y1 - node.y0, node.x1 - node.x0
            if height != width or height not in {1, 2, 4, 8}:
                raise ValueError("native multiscale topology contains a non-native node")
            area_scale = height * width
            _, _, grids = scale_sources[area_scale]
            expected_grids = tuple((h // height, w // width) for h, w in forest.grids)
            if grids != expected_grids:
                raise ValueError(
                    f"scale {area_scale} grids {grids} do not match expected {expected_grids}"
                )
            offset = sum(h * w for h, w in grids[: node.image_index])
            grid_width = grids[node.image_index][1]
            source_index = offset + (node.y0 // height) * grid_width + node.x0 // width
            node_ids_by_scale[area_scale].append(node.node_id)
            source_indices_by_scale[area_scale].append(source_index)

        for area_scale in sorted(required):
            source_keys, source_values, _ = scale_sources[area_scale]
            if source_keys.device != raw_keys.device or source_values.device != values.device:
                raise ValueError("all native scale tensors must be on the same device")
            node_ids = torch.tensor(node_ids_by_scale[area_scale], dtype=torch.long, device=raw_keys.device)
            indices = torch.tensor(
                source_indices_by_scale[area_scale], dtype=torch.long, device=raw_keys.device
            )
            raw_keys.index_copy_(-2, node_ids, source_keys.index_select(-2, indices))
            values.index_copy_(-2, node_ids, source_values.index_select(-2, indices))
        return cls(
            raw_keys=raw_keys,
            values=values,
            native_positions=_pool_nodes(fine_positions, topology, -1),
        )

    @classmethod
    def from_hidden(
        cls,
        topology: DeviceTreeTopology,
        fine_hidden: torch.Tensor,
        fine_positions: torch.Tensor,
        projector: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    ) -> LayerKVPyramid:
        if fine_hidden.ndim != 3 or fine_hidden.shape[1] != topology.leaf_nodes.numel():
            raise ValueError("fine_hidden must have shape [batch, leaves, hidden_size]")
        hidden_nodes = _pool_nodes(fine_hidden, topology, 1)
        raw_keys, values = projector(hidden_nodes)
        return cls(
            raw_keys=raw_keys,
            values=values,
            native_positions=_pool_nodes(fine_positions, topology, -1),
        )

    @classmethod
    def from_kv(
        cls,
        topology: DeviceTreeTopology,
        fine_raw_keys: torch.Tensor,
        fine_values: torch.Tensor,
        fine_positions: torch.Tensor,
        fine_rotated_keys: torch.Tensor | None = None,
    ) -> LayerKVPyramid:
        if fine_raw_keys.ndim != 4 or fine_values.shape != fine_raw_keys.shape:
            raise ValueError("fine K/V must have shape [batch, kv_heads, leaves, head_dim]")
        if fine_raw_keys.shape[-2] != topology.leaf_nodes.numel():
            raise ValueError("fine K/V count does not match the spatial forest")
        if fine_positions.ndim != 3 or fine_positions.shape[0] != 3:
            raise ValueError("fine_positions must have shape [3, batch, leaves]")
        return cls(
            raw_keys=_pool_nodes(fine_raw_keys, topology, -2),
            values=_pool_nodes(fine_values, topology, -2),
            native_positions=_pool_nodes(fine_positions, topology, -1),
            post_rope_keys=(_pool_nodes(fine_rotated_keys, topology, -2) if fine_rotated_keys is not None else None),
        )

    def gather(
        self,
        active_ids: torch.Tensor,
        position_mode: str,
        rotate_key: RotateKey,
        position_encoder: PositionEncoder,
        reference: torch.Tensor,
        anchor_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = self.values.index_select(-2, active_ids)
        if position_mode == "post_rope_pool":
            if self.post_rope_keys is None:
                raise RuntimeError("post-RoPE keys were not captured")
            return self.post_rope_keys.index_select(-2, active_ids), values

        raw_keys = self.raw_keys.index_select(-2, active_ids)
        if position_mode == "no_rope":
            return raw_keys, values
        if position_mode == "text_anchor":
            if anchor_positions is None:
                raise RuntimeError("text_anchor requires anchor positions")
            positions = anchor_positions
        else:
            positions = self.native_positions.index_select(-1, active_ids)
        cos, sin = position_encoder(reference, positions)
        return rotate_key(raw_keys, cos, sin), values
