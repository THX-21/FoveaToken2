from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True, slots=True)
class BlockNode:
    y0: int
    x0: int
    size: int
    leaves: tuple[int, ...]


@dataclass(slots=True)
class BlockFront:
    height: int
    width: int
    nodes: tuple[BlockNode, ...]
    scale_counts: dict[int, int]
    _tensor_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = field(
        default_factory=dict, init=False, repr=False
    )

    @property
    def leaf_count(self) -> int:
        return self.height * self.width

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def compression_ratio(self) -> float:
        return self.node_count / self.leaf_count

    @classmethod
    def uniform(cls, height: int, width: int, block_size: int) -> "BlockFront":
        _validate_grid(height, width)
        if block_size not in {1, 2, 4, 8}:
            raise ValueError("block_size must be 1, 2, 4, or 8")
        if height % block_size or width % block_size:
            raise ValueError("grid is not divisible by block_size")
        nodes = tuple(
            _node(height, width, y, x, block_size)
            for y in range(0, height, block_size)
            for x in range(0, width, block_size)
        )
        front = cls(height, width, nodes, {block_size: height * width // (block_size**2)})
        front.validate()
        return front

    @classmethod
    def random_multiscale(
        cls,
        height: int,
        width: int,
        seed: int,
        area_ratios: tuple[float, float, float] = (0.50, 0.30, 0.20),
    ) -> "BlockFront":
        _validate_grid(height, width)
        if height % 4 or width % 4:
            raise ValueError("random multiscale grids must be divisible by 4")
        if len(area_ratios) != 3 or any(value < 0 for value in area_ratios):
            raise ValueError("area_ratios must contain three non-negative values")
        if not math.isclose(sum(area_ratios), 1.0, abs_tol=1e-9):
            raise ValueError("area_ratios must sum to one")
        macroblocks = [(y, x) for y in range(0, height, 4) for x in range(0, width, 4)]
        counts = _largest_remainder(len(macroblocks), area_ratios)
        scales = [1] * counts[0] + [2] * counts[1] + [4] * counts[2]
        rng = random.Random(seed)
        rng.shuffle(scales)
        nodes: list[BlockNode] = []
        scale_counts = {1: 0, 2: 0, 4: 0}
        for (macro_y, macro_x), scale in zip(macroblocks, scales):
            for y in range(macro_y, macro_y + 4, scale):
                for x in range(macro_x, macro_x + 4, scale):
                    nodes.append(_node(height, width, y, x, scale))
                    scale_counts[scale] += 1
        front = cls(height, width, tuple(nodes), scale_counts)
        front.validate()
        return front

    def validate(self) -> None:
        covered = [leaf for node in self.nodes for leaf in node.leaves]
        if len(covered) != self.leaf_count or sorted(covered) != list(range(self.leaf_count)):
            raise ValueError("front nodes must form a non-overlapping cover of the visual grid")
        for node in self.nodes:
            if node.size not in {1, 2, 4, 8} or len(node.leaves) != node.size**2:
                raise ValueError("front contains an invalid block node")

    def pool(self, fine: torch.Tensor, node_dim: int) -> torch.Tensor:
        node_dim %= fine.ndim
        if fine.shape[node_dim] != self.leaf_count:
            raise ValueError("fine tensor does not match front leaf count")
        leaves, groups, weights = self._indices(fine.device)
        ordered = fine.index_select(node_dim, leaves).movedim(node_dim, 0)
        weight_shape = (weights.numel(),) + (1,) * (ordered.ndim - 1)
        output = fine.new_zeros((self.node_count, *ordered.shape[1:]))
        output.index_add_(0, groups, ordered * weights.to(fine.dtype).view(weight_shape))
        return output.movedim(0, node_dim)

    def _indices(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = torch.device(device)
        cached = self._tensor_cache.get(device)
        if cached is not None:
            return cached
        leaves = torch.tensor(
            [leaf for node in self.nodes for leaf in node.leaves], dtype=torch.long, device=device
        )
        groups = torch.tensor(
            [group for group, node in enumerate(self.nodes) for _ in node.leaves],
            dtype=torch.long,
            device=device,
        )
        weights = torch.tensor(
            [1.0 / len(node.leaves) for node in self.nodes for _ in node.leaves],
            dtype=torch.float32,
            device=device,
        )
        self._tensor_cache[device] = (leaves, groups, weights)
        return leaves, groups, weights

    def digest(self) -> str:
        payload = ";".join(
            f"{node.y0},{node.x0},{node.size}" for node in self.nodes
        ).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


def stable_seed(base_seed: int, sample_id: str, step: int | None = None) -> int:
    parts = [str(base_seed), sample_id]
    if step is not None:
        parts.append(str(step))
    digest = hashlib.sha256("\0".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _node(height: int, width: int, y: int, x: int, size: int) -> BlockNode:
    if y + size > height or x + size > width:
        raise ValueError("block exceeds visual grid")
    leaves = tuple((row * width + column) for row in range(y, y + size) for column in range(x, x + size))
    return BlockNode(y, x, size, leaves)


def _largest_remainder(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    raw = [total * ratio for ratio in ratios]
    counts = [math.floor(value) for value in raw]
    remaining = total - sum(counts)
    order = sorted(range(3), key=lambda index: (raw[index] - counts[index], -index), reverse=True)
    for index in order[:remaining]:
        counts[index] += 1
    return counts[0], counts[1], counts[2]


def _validate_grid(height: int, width: int) -> None:
    if height <= 0 or width <= 0:
        raise ValueError("visual grid dimensions must be positive")
