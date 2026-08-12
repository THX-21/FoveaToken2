from __future__ import annotations

from typing import Literal

import torch

from .topology import DeviceTreeTopology


class SplitMergeRouter:
    """Fixed-budget Split-Merge routing with device-resident state."""

    def __init__(
        self,
        topology: DeviceTreeTopology,
        initial_active: torch.Tensor,
        epsilon: float = 0.05,
        max_swaps: int = 8,
        score_mode: Literal["mass", "density"] = "mass",
    ):
        if score_mode not in {"mass", "density"}:
            raise ValueError(f"unsupported score_mode: {score_mode}")
        self.topology = topology
        self.epsilon = epsilon
        self.max_swaps = max_swaps
        self.score_mode = score_mode
        self.budget = initial_active.numel()
        self.active = torch.zeros(topology.node_count, dtype=torch.bool, device=topology.device)
        self.active.index_fill_(0, initial_active, True)
        self._node_ids = torch.arange(topology.node_count, device=topology.device)

    def active_ids(self) -> torch.Tensor:
        # The budget is fixed, so topk avoids CUDA's dynamic-shape nonzero synchronization.
        return torch.topk(self.active.to(torch.uint8), self.budget, sorted=False).indices

    def scores_from_active(
        self, active_ids: torch.Tensor, active_scores: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        leaf_scores = self.topology.project_active_to_leaves(self.active, active_ids, active_scores)
        node_scores = self.topology.aggregate_leaves(leaf_scores, density=self.score_mode == "density")
        return node_scores, leaf_scores

    def step(self, scores: torch.Tensor) -> torch.Tensor:
        """Update the front without copying scores or routing state to the CPU."""
        if scores.ndim != 1 or scores.numel() != self.topology.node_count:
            raise ValueError("scores must contain one value per spatial node")
        if self.topology.node_count < 2 or self.max_swaps == 0:
            return torch.zeros((), dtype=torch.long, device=scores.device)
        topology = self.topology
        enabled = torch.ones((), dtype=torch.bool, device=scores.device)
        swaps = torch.zeros((), dtype=torch.long, device=scores.device)
        infinity = torch.tensor(torch.inf, dtype=scores.dtype, device=scores.device)

        for _ in range(self.max_swaps):
            split_mask = self.active & (topology.child_count > 0)
            child_active = self.active.index_select(0, topology.children.reshape(-1)).view_as(topology.children)
            merge_mask = (topology.child_count > 0) & (child_active | ~topology.child_mask).all(dim=-1)

            chosen_merge_score = torch.full_like(scores, torch.inf)
            chosen_merge_id = torch.zeros_like(topology.parent)
            split_delta = topology.child_count - 1
            for delta in (1, 2, 3):
                candidates = merge_mask & (split_delta == delta)
                costs = torch.where(candidates, scores, infinity)
                values, ids = torch.topk(costs, k=2, largest=False, sorted=True)
                same_parent = topology.parent == ids[0]
                selected_score = torch.where(same_parent, values[1], values[0])
                selected_id = torch.where(same_parent, ids[1], ids[0])
                matching_split = split_mask & (split_delta == delta)
                chosen_merge_score = torch.where(matching_split, selected_score, chosen_merge_score)
                chosen_merge_id = torch.where(matching_split, selected_id, chosen_merge_id)

            advantage = scores - chosen_merge_score
            legal_split = split_mask & torch.isfinite(chosen_merge_score)
            best_split = torch.argmax(torch.where(legal_split, advantage, -infinity))
            best_merge = chosen_merge_id[best_split]
            legal = legal_split[best_split] & (
                scores[best_split] > (1.0 + self.epsilon) * chosen_merge_score[best_split]
            )
            apply_swap = enabled & legal

            split_children = topology.children[best_split]
            split_child_mask = topology.child_mask[best_split]
            merge_children = topology.children[best_merge]
            merge_child_mask = topology.child_mask[best_merge]
            remove_mask = (self._node_ids == best_split) | (
                (self._node_ids[:, None] == merge_children[None, :]) & merge_child_mask[None, :]
            ).any(dim=-1)
            add_mask = (self._node_ids == best_merge) | (
                (self._node_ids[:, None] == split_children[None, :]) & split_child_mask[None, :]
            ).any(dim=-1)
            self.active = torch.where(apply_swap & remove_mask, False, self.active)
            self.active = torch.where(apply_swap & add_mask, True, self.active)
            swaps = swaps + apply_swap.to(torch.long)
            enabled = apply_swap
        return swaps
