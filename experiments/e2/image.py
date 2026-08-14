from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True, slots=True)
class ImagePlan:
    width: int
    height: int
    grid_width: int
    grid_height: int

    @property
    def visual_tokens(self) -> int:
        return self.grid_width * self.grid_height


def aligned_high_resolution(
    image: Image.Image, pixel_per_token: int, min_pixels: int, max_pixels: int
) -> ImagePlan:
    width, height = image.size
    area = width * height
    scale = math.sqrt(min(max(area, min_pixels), max_pixels) / area)
    target_width, target_height = width * scale, height * scale
    unit = pixel_per_token * 4
    grid_width = max(4, round(target_width / unit) * 4)
    grid_height = max(4, round(target_height / unit) * 4)
    return ImagePlan(
        grid_width * pixel_per_token,
        grid_height * pixel_per_token,
        grid_width,
        grid_height,
    )


def lowres_plan(high: ImagePlan, divisor: int) -> ImagePlan:
    if divisor not in {2, 4, 8}:
        raise ValueError("low-resolution divisor must be 2, 4, or 8")
    if high.grid_width % divisor or high.grid_height % divisor:
        raise ValueError("high-resolution grid is not divisible by low-resolution divisor")
    return ImagePlan(
        high.width // divisor,
        high.height // divisor,
        high.grid_width // divisor,
        high.grid_height // divisor,
    )


def matched_lowres_plan(high: ImagePlan, target_tokens: int, pixel_per_token: int) -> ImagePlan:
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    aspect = high.grid_width / high.grid_height
    best: tuple[float, int, int] | None = None
    for height in range(1, high.grid_height + 1):
        width = max(1, round(target_tokens / height))
        for candidate_width in {max(1, width - 1), width, width + 1}:
            if candidate_width > high.grid_width:
                continue
            token_error = abs(candidate_width * height - target_tokens)
            aspect_error = abs(math.log((candidate_width / height) / aspect))
            score = token_error * 1000 + aspect_error
            if best is None or score < best[0]:
                best = (score, candidate_width, height)
    assert best is not None
    grid_width, grid_height = best[1], best[2]
    return ImagePlan(
        grid_width * pixel_per_token,
        grid_height * pixel_per_token,
        grid_width,
        grid_height,
    )


def resize(image: Image.Image, plan: ImagePlan) -> Image.Image:
    return image.convert("RGB").resize((plan.width, plan.height), Image.Resampling.LANCZOS)
