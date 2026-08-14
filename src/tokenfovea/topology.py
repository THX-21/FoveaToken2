from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class SpatialNode:
    node_id: int
    parent_id: int | None
    children: tuple[int, ...]
    image_index: int
    y0: int
    x0: int
    y1: int
    x1: int
    depth: int
    leaf_indices: tuple[int, ...]

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def valid_count(self) -> int:
        return len(self.leaf_indices)

    @property
    def split_delta(self) -> int:
        return max(0, len(self.children) - 1)


class VisualTokenForest:
    """Ragged quadtree forest for arbitrary rectangular visual grids."""

    def __init__(
        self,
        nodes: list[SpatialNode],
        roots: tuple[int, ...],
        grids: tuple[tuple[int, int], ...],
    ):
        self.nodes = nodes
        self.roots = roots
        self.grids = grids
        self.num_leaves = sum(h * w for h, w in grids)

    @classmethod
    def from_grids(cls, grids: Iterable[tuple[int, int]]) -> VisualTokenForest:
        grid_tuple = tuple((int(h), int(w)) for h, w in grids)
        if not grid_tuple or any(h <= 0 or w <= 0 for h, w in grid_tuple):
            raise ValueError("grids must contain positive (height, width) pairs")

        drafts: list[dict] = []
        roots: list[int] = []
        leaf_offset = 0

        def build(image_index: int, width: int, y0: int, x0: int, y1: int, x1: int, depth: int) -> int:
            if y1 - y0 == 1 and x1 - x0 == 1:
                node_id = len(drafts)
                drafts.append(
                    {
                        "node_id": node_id,
                        "parent_id": None,
                        "children": (),
                        "image_index": image_index,
                        "y0": y0,
                        "x0": x0,
                        "y1": y1,
                        "x1": x1,
                        "depth": depth,
                        "leaf_indices": (leaf_offset + y0 * width + x0,),
                    }
                )
                return node_id

            ym = y0 + (y1 - y0 + 1) // 2
            xm = x0 + (x1 - x0 + 1) // 2
            boxes = (
                (y0, x0, ym, xm),
                (y0, xm, ym, x1),
                (ym, x0, y1, xm),
                (ym, xm, y1, x1),
            )
            children = tuple(
                build(image_index, width, cy0, cx0, cy1, cx1, depth + 1)
                for cy0, cx0, cy1, cx1 in boxes
                if cy0 < cy1 and cx0 < cx1
            )
            node_id = len(drafts)
            drafts.append(
                {
                    "node_id": node_id,
                    "parent_id": None,
                    "children": children,
                    "image_index": image_index,
                    "y0": y0,
                    "x0": x0,
                    "y1": y1,
                    "x1": x1,
                    "depth": depth,
                    "leaf_indices": tuple(i for child in children for i in drafts[child]["leaf_indices"]),
                }
            )
            for child in children:
                drafts[child]["parent_id"] = node_id
            return node_id

        for image_index, (height, width) in enumerate(grid_tuple):
            roots.append(build(image_index, width, 0, 0, height, width, 0))
            leaf_offset += height * width
        return cls([SpatialNode(**draft) for draft in drafts], tuple(roots), grid_tuple)

    @classmethod
    def from_aligned_grids(
        cls,
        grids: Iterable[tuple[int, int]],
        *,
        max_block_size: int = 8,
    ) -> VisualTokenForest:
        """Build independent exact quadtrees over aligned square macroblocks.

        Unlike :meth:`from_grids`, every node is an aligned square with edge
        length 1, 2, 4, ..., ``max_block_size``.  Native multiscale banks rely
        on that one-to-one mapping between tree nodes and native resolutions.
        """
        grid_tuple = tuple((int(h), int(w)) for h, w in grids)
        if not grid_tuple or any(h <= 0 or w <= 0 for h, w in grid_tuple):
            raise ValueError("grids must contain positive (height, width) pairs")
        if max_block_size <= 0 or max_block_size & (max_block_size - 1):
            raise ValueError("max_block_size must be a positive power of two")
        if any(h % max_block_size or w % max_block_size for h, w in grid_tuple):
            raise ValueError(
                f"native multiscale grids must be divisible by {max_block_size}"
            )

        drafts: list[dict] = []
        roots: list[int] = []
        leaf_offset = 0

        def build(
            image_index: int,
            width: int,
            y0: int,
            x0: int,
            size: int,
            depth: int,
        ) -> int:
            if size == 1:
                node_id = len(drafts)
                drafts.append(
                    {
                        "node_id": node_id,
                        "parent_id": None,
                        "children": (),
                        "image_index": image_index,
                        "y0": y0,
                        "x0": x0,
                        "y1": y0 + 1,
                        "x1": x0 + 1,
                        "depth": depth,
                        "leaf_indices": (leaf_offset + y0 * width + x0,),
                    }
                )
                return node_id
            half = size // 2
            children = tuple(
                build(image_index, width, y0 + dy, x0 + dx, half, depth + 1)
                for dy, dx in ((0, 0), (0, half), (half, 0), (half, half))
            )
            node_id = len(drafts)
            drafts.append(
                {
                    "node_id": node_id,
                    "parent_id": None,
                    "children": children,
                    "image_index": image_index,
                    "y0": y0,
                    "x0": x0,
                    "y1": y0 + size,
                    "x1": x0 + size,
                    "depth": depth,
                    "leaf_indices": tuple(i for child in children for i in drafts[child]["leaf_indices"]),
                }
            )
            for child in children:
                drafts[child]["parent_id"] = node_id
            return node_id

        for image_index, (height, width) in enumerate(grid_tuple):
            for y0 in range(0, height, max_block_size):
                for x0 in range(0, width, max_block_size):
                    roots.append(build(image_index, width, y0, x0, max_block_size, 0))
            leaf_offset += height * width
        forest = cls([SpatialNode(**draft) for draft in drafts], tuple(roots), grid_tuple)
        forest.validate_front(forest.roots)
        return forest

    def node(self, node_id: int) -> SpatialNode:
        return self.nodes[node_id]

    def initial_front(self, target_budget: int) -> set[int]:
        """Create the closest reachable spatially uniform front."""
        if target_budget <= 0:
            raise ValueError("target_budget must be positive")
        active = set(self.roots)
        target_budget = min(max(target_budget, len(active)), self.num_leaves)
        while True:
            candidates = [self.node(i) for i in active if self.node(i).children]
            if not candidates:
                break
            best = min(
                candidates,
                key=lambda node: (
                    abs(len(active) + node.split_delta - target_budget),
                    node.depth,
                    node.image_index,
                    node.node_id,
                ),
            )
            new_size = len(active) + best.split_delta
            if abs(new_size - target_budget) >= abs(len(active) - target_budget):
                break
            active.remove(best.node_id)
            active.update(best.children)
        self.validate_front(active)
        return active

    def validate_front(self, active: Iterable[int]) -> None:
        covered = [leaf for node_id in set(active) for leaf in self.node(node_id).leaf_indices]
        if sorted(covered) != list(range(self.num_leaves)):
            raise AssertionError("active nodes must form a non-overlapping cover of all visual leaves")


@dataclass(slots=True)
class DeviceTreeTopology:
    """Static tree topology encoded once as device tensors."""

    children: torch.Tensor
    child_mask: torch.Tensor
    child_count: torch.Tensor
    parent: torch.Tensor
    valid_count: torch.Tensor
    leaf_nodes: torch.Tensor
    ancestors: torch.Tensor
    ancestor_mask: torch.Tensor
    normalized_centers: torch.Tensor
    levels: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]

    @classmethod
    def build(cls, forest: VisualTokenForest, device: torch.device) -> DeviceTreeTopology:
        node_count = len(forest.nodes)
        children_data = [[0, 0, 0, 0] for _ in forest.nodes]
        child_mask_data = [[False, False, False, False] for _ in forest.nodes]
        child_count_data = [0] * node_count
        parent_data = [-1] * node_count
        valid_count = torch.tensor(
            [node.valid_count for node in forest.nodes],
            dtype=torch.float32,
            device=device,
        )

        leaf_node_ids = [0] * forest.num_leaves
        for node in forest.nodes:
            if node.children:
                count = len(node.children)
                children_data[node.node_id][:count] = node.children
                child_mask_data[node.node_id][:count] = [True] * count
                child_count_data[node.node_id] = count
            else:
                leaf_node_ids[node.leaf_indices[0]] = node.node_id
            if node.parent_id is not None:
                parent_data[node.node_id] = node.parent_id

        children = torch.tensor(children_data, dtype=torch.long, device=device)
        child_mask = torch.tensor(child_mask_data, dtype=torch.bool, device=device)
        child_count = torch.tensor(child_count_data, dtype=torch.long, device=device)
        parent = torch.tensor(parent_data, dtype=torch.long, device=device)

        ancestor_lists = []
        for leaf_node_id in leaf_node_ids:
            lineage = []
            node_id: int | None = leaf_node_id
            while node_id is not None:
                lineage.append(node_id)
                node_id = forest.node(node_id).parent_id
            ancestor_lists.append(lineage)
        max_ancestors = max(map(len, ancestor_lists))
        ancestors = torch.tensor(
            [lineage + [0] * (max_ancestors - len(lineage)) for lineage in ancestor_lists],
            dtype=torch.long,
            device=device,
        )
        ancestor_mask = torch.tensor(
            [[True] * len(lineage) + [False] * (max_ancestors - len(lineage)) for lineage in ancestor_lists],
            dtype=torch.bool,
            device=device,
        )
        normalized_centers = torch.tensor(
            [
                (
                    (node.y0 + node.y1) / (2.0 * forest.grids[node.image_index][0]),
                    (node.x0 + node.x1) / (2.0 * forest.grids[node.image_index][1]),
                )
                for node in forest.nodes
            ],
            dtype=torch.float32,
            device=device,
        )

        levels = []
        internal_depths = sorted({node.depth for node in forest.nodes if node.children}, reverse=True)
        for depth in internal_depths:
            ids = [node.node_id for node in forest.nodes if node.children and node.depth == depth]
            parent_ids = torch.tensor(ids, dtype=torch.long, device=device)
            level_children = children.index_select(0, parent_ids)
            level_mask = child_mask.index_select(0, parent_ids)
            counts = valid_count.index_select(0, level_children.reshape(-1)).view_as(level_children)
            weights = counts * level_mask
            weights = weights / weights.sum(dim=-1, keepdim=True)
            levels.append((parent_ids, level_children, weights))

        return cls(
            children=children,
            child_mask=child_mask,
            child_count=child_count,
            parent=parent,
            valid_count=valid_count,
            leaf_nodes=torch.tensor(leaf_node_ids, dtype=torch.long, device=device),
            ancestors=ancestors,
            ancestor_mask=ancestor_mask,
            normalized_centers=normalized_centers,
            levels=tuple(levels),
        )

    @property
    def device(self) -> torch.device:
        return self.children.device

    @property
    def node_count(self) -> int:
        return self.children.shape[0]

    def aggregate_leaves(self, leaf_scores: torch.Tensor, *, density: bool) -> torch.Tensor:
        """Aggregate leaf scores to every node using a few level-wise kernels."""
        scores = leaf_scores.new_zeros(self.node_count)
        scores.index_copy_(0, self.leaf_nodes, leaf_scores)
        for parent_ids, child_ids, weights in self.levels:
            child_scores = scores.index_select(0, child_ids.reshape(-1)).view_as(child_ids)
            pooled = (
                (child_scores * weights.to(child_scores.dtype)).sum(dim=-1) if density else child_scores.sum(dim=-1)
            )
            scores.index_copy_(0, parent_ids, pooled)
        return scores

    def project_active_to_leaves(
        self,
        active: torch.Tensor,
        active_ids: torch.Tensor,
        active_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Uniformly distribute every active node's mass over its covered leaves."""
        node_scores = active_scores.new_zeros(self.node_count)
        node_scores.index_copy_(0, active_ids, active_scores)
        ancestor_scores = node_scores.index_select(0, self.ancestors.reshape(-1)).view_as(self.ancestors)
        ancestor_sizes = self.valid_count.index_select(0, self.ancestors.reshape(-1)).view_as(self.ancestors)
        active_ancestors = active.index_select(0, self.ancestors.reshape(-1)).view_as(self.ancestors)
        mask = self.ancestor_mask & active_ancestors
        return ((ancestor_scores / ancestor_sizes) * mask.to(ancestor_scores.dtype)).sum(dim=-1)
