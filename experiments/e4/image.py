from __future__ import annotations

import math

from PIL import Image

from tokenfovea.multiscale import NativeImagePlan


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


def resize(image: Image.Image, plan: NativeImagePlan) -> Image.Image:
    return image.convert("RGB").resize((plan.width, plan.height), Image.Resampling.LANCZOS)


def visual_tokens(plan: NativeImagePlan) -> int:
    return plan.grid_width * plan.grid_height
