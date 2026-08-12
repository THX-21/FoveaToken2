from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class HybridStatistics:
    coverage: float
    persistence: float


@dataclass(frozen=True, slots=True)
class GazeStatistics:
    raw_score: float
    null_score: float
    calibrated_score: float


def visual_statistics(visual_attention: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return visual mass, concentration, and the visual-normalized distribution.

    ``visual_attention`` contains post-softmax mass from the complete attention
    context and has shape ``[heads, visual_tokens]``.
    """

    if visual_attention.ndim != 2 or visual_attention.shape[-1] == 0:
        raise ValueError("visual attention must have shape [heads, nonzero_visual_tokens]")
    attention = visual_attention.float().clamp_min(0)
    mass = attention.sum(dim=-1)
    distribution = attention / mass[:, None].clamp_min(1e-12)
    token_count = distribution.shape[-1]
    if token_count == 1:
        concentration = torch.ones_like(mass)
    else:
        entropy = -(distribution * distribution.clamp_min(1e-12).log()).sum(dim=-1)
        concentration = 1.0 - entropy / math.log(token_count)
    return mass, concentration.clamp(0, 1), distribution


def hybrid_statistics(topk_counts: torch.Tensor, step_count: int) -> list[HybridStatistics]:
    """Compute visual-only HybridKV coverage and persistence for each head."""

    if topk_counts.ndim != 2 or topk_counts.shape[-1] == 0:
        raise ValueError("topk_counts must have shape [heads, visual_tokens]")
    if step_count <= 0:
        raise ValueError("step_count must be positive")
    coverage = (topk_counts > 0).float().mean(dim=-1)
    persistence = topk_counts.max(dim=-1).values.float() / step_count
    return [
        HybridStatistics(float(head_coverage), float(head_persistence))
        for head_coverage, head_persistence in zip(coverage, persistence)
    ]


def gaze_statistics(matrix: torch.Tensor, null_vector: torch.Tensor) -> list[GazeStatistics]:
    """Score per-head K-by-K gaze matrices against their null-prompt baseline."""

    if matrix.ndim != 3 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("gaze matrix must have shape [heads, regions, regions]")
    if null_vector.shape != (matrix.shape[0], matrix.shape[-1]):
        raise ValueError("null vector must have shape [heads, regions]")
    raw = matrix.diagonal(dim1=-2, dim2=-1).mean(dim=-1)
    null = null_vector.mean(dim=-1)
    calibrated = raw - null
    return [
        GazeStatistics(float(head_raw), float(head_null), float(head_calibrated))
        for head_raw, head_null, head_calibrated in zip(raw, null, calibrated)
    ]
