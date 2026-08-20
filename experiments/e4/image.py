from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import Image

from tokenfovea.multiscale import NativeImagePlan


@dataclass(frozen=True, slots=True)
class MatchedBudgetPlan:
    """Low-resolution grids sharing one exact budget with a Native front."""

    lowres_plans: tuple[NativeImagePlan, ...]
    requested_ratio: float
    theoretical_tokens: float
    target_tokens: int
    achieved_ratio: float
    relative_budget_error: float
    max_aspect_log_error: float


def aligned_high_resolution(
    image: Image.Image,
    pixel_per_token: int,
    min_pixels: int,
    token_cap: int,
) -> NativeImagePlan:
    if token_cap <= 0 or pixel_per_token <= 0:
        raise ValueError("invalid E4 image limits")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    aspect = width / height
    raw_tokens = width * height / pixel_per_token**2
    minimum = max(64, math.ceil(min_pixels / pixel_per_token**2 / 64) * 64)
    target = min(token_cap, max(minimum, round(raw_tokens)))
    best: tuple[float, int, int] | None = None
    for grid_height in range(8, token_cap // 8 + 1, 8):
        ideal_width = target / grid_height
        base = max(8, round(ideal_width / 8) * 8)
        for grid_width in {max(8, base - 8), base, base + 8}:
            tokens = grid_height * grid_width
            if tokens > token_cap:
                continue
            ratio_error = abs(math.log((grid_width / grid_height) / aspect))
            area_error = abs(tokens - target) / target
            score = ratio_error + area_error
            if best is None or score < best[0]:
                best = (score, grid_width, grid_height)
    if best is None:
        raise ValueError("cannot build an 8-aligned E4 image grid")
    grid_width, grid_height = best[1], best[2]
    return NativeImagePlan(
        width=grid_width * pixel_per_token,
        height=grid_height * pixel_per_token,
        grid_width=grid_width,
        grid_height=grid_height,
    )


def scaled_plan(plan: NativeImagePlan, divisor: int) -> NativeImagePlan:
    if divisor not in {1, 2, 4, 8}:
        raise ValueError("E4 scale divisor must be 1, 2, 4, or 8")
    if plan.grid_width % divisor or plan.grid_height % divisor:
        raise ValueError("E4 high-resolution grid is not divisible by divisor")
    return NativeImagePlan(
        width=plan.width // divisor,
        height=plan.height // divisor,
        grid_width=plan.grid_width // divisor,
        grid_height=plan.grid_height // divisor,
    )


def matched_budget_plan(
    plans: list[NativeImagePlan],
    compression_ratio: float,
) -> MatchedBudgetPlan:
    """Choose aspect-preserving LowRes grids at a Native-reachable budget.

    Aligned Native quadtrees have one root per 8x8 block and every split adds
    three active nodes. Therefore an exact front budget ``B`` must satisfy
    ``B == roots (mod 3)``. LowRes grids are searched jointly so their actual
    token count is exactly that same ``B`` rather than merely rounded to it.
    """
    if not plans:
        raise ValueError("matched budget planning requires at least one image")
    if not math.isfinite(compression_ratio) or not 1.0 <= compression_ratio <= 64.0:
        raise ValueError("compression_ratio must be finite and between 1 and 64")
    if any(plan.grid_height % 8 or plan.grid_width % 8 for plan in plans):
        raise ValueError("matched budget planning requires 8-aligned high-resolution grids")

    high_tokens = sum(visual_tokens(plan) for plan in plans)
    theoretical = high_tokens / compression_ratio
    roots = sum(visual_tokens(plan) // 64 for plan in plans)
    candidates = [_lowres_candidates(plan, compression_ratio) for plan in plans]

    states: list[tuple[float, int, tuple[NativeImagePlan, ...]]] = [(0.0, 0, ())]
    partial_target = 0.0
    for plan, image_candidates in zip(plans, candidates):
        partial_target += visual_tokens(plan) / compression_ratio
        expanded = [
            (score + candidate_score, tokens + candidate_tokens, chosen + (candidate,))
            for score, tokens, chosen in states
            for candidate_score, candidate_tokens, candidate in image_candidates
        ]
        ranked: dict[int, list[tuple[float, int, tuple[NativeImagePlan, ...]]]] = {
            0: [],
            1: [],
            2: [],
        }
        for state in expanded:
            ranked[state[1] % 3].append(state)
        states = []
        for residue in range(3):
            ranked[residue].sort(
                key=lambda state: (
                    abs(state[1] - partial_target) / max(partial_target, 1.0)
                    + state[0] / len(state[2]),
                    state[1],
                )
            )
            states.extend(ranked[residue][:128])

    feasible = [
        state
        for state in states
        if roots <= state[1] <= high_tokens and (state[1] - roots) % 3 == 0
    ]
    if not feasible:
        raise ValueError("cannot find a LowRes grid with a Native-reachable budget")
    _, target_tokens, lowres_plans = min(
        feasible,
        key=lambda state: (
            abs(state[1] - theoretical) / theoretical + state[0] / len(plans),
            abs(state[1] - theoretical),
            state[1],
        ),
    )
    return MatchedBudgetPlan(
        lowres_plans=lowres_plans,
        requested_ratio=float(compression_ratio),
        theoretical_tokens=theoretical,
        target_tokens=target_tokens,
        achieved_ratio=high_tokens / target_tokens,
        relative_budget_error=abs(target_tokens - theoretical) / theoretical,
        max_aspect_log_error=max(
            abs(
                math.log(
                    (lowres.grid_width / lowres.grid_height)
                    / (highres.grid_width / highres.grid_height)
                )
            )
            for highres, lowres in zip(plans, lowres_plans)
        ),
    )


def _lowres_candidates(
    plan: NativeImagePlan,
    compression_ratio: float,
) -> list[tuple[float, int, NativeImagePlan]]:
    high_height, high_width = plan.grid_height, plan.grid_width
    target = high_height * high_width / compression_ratio
    aspect = high_width / high_height
    pixel_per_token_w = plan.width // high_width
    pixel_per_token_h = plan.height // high_height
    candidates: dict[tuple[int, int], tuple[float, int, NativeImagePlan]] = {}
    for height in range(1, high_height + 1):
        area_width = target / height
        aspect_width = height * aspect
        width_values: set[int] = set()
        for center in (area_width, aspect_width, math.sqrt(target * aspect)):
            rounded = round(center)
            width_values.update(range(rounded - 2, rounded + 3))
        for width in width_values:
            if not 1 <= width <= high_width:
                continue
            tokens = height * width
            area_error = abs(tokens - target) / target
            aspect_error = abs(math.log((width / height) / aspect))
            score = area_error + aspect_error
            candidate = NativeImagePlan(
                width=width * pixel_per_token_w,
                height=height * pixel_per_token_h,
                grid_width=width,
                grid_height=height,
            )
            candidates[(height, width)] = (score, tokens, candidate)
    ranked = sorted(
        candidates.values(),
        key=lambda item: (
            item[0],
            abs(item[1] - target),
            item[2].grid_height,
            item[2].grid_width,
        ),
    )
    compact: list[tuple[float, int, NativeImagePlan]] = []
    for residue in range(3):
        compact.extend([item for item in ranked if item[1] % 3 == residue][:32])
    return compact


def resize(image: Image.Image, plan: NativeImagePlan) -> Image.Image:
    return image.convert("RGB").resize((plan.width, plan.height), Image.Resampling.LANCZOS)


def visual_tokens(plan: NativeImagePlan) -> int:
    return plan.grid_width * plan.grid_height
