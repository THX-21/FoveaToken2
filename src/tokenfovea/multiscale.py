from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from PIL import Image


@dataclass(frozen=True, slots=True)
class NativeImagePlan:
    width: int
    height: int
    grid_width: int
    grid_height: int


def aligned_native_plan(
    image: Image.Image,
    pixel_per_token: int,
    min_pixels: int,
    max_pixels: int,
    *,
    grid_multiple: int = 8,
) -> NativeImagePlan:
    """Choose an aspect-preserving LLM-token grid aligned for native scales."""
    if pixel_per_token <= 0 or min_pixels <= 0 or max_pixels < min_pixels:
        raise ValueError("invalid native image sizing parameters")
    if grid_multiple <= 0:
        raise ValueError("grid_multiple must be positive")
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    area = width * height
    scale = math.sqrt(min(max(area, min_pixels), max_pixels) / area)
    target_width, target_height = width * scale, height * scale
    unit = pixel_per_token * grid_multiple
    grid_width = max(grid_multiple, round(target_width / unit) * grid_multiple)
    grid_height = max(grid_multiple, round(target_height / unit) * grid_multiple)
    return NativeImagePlan(
        width=grid_width * pixel_per_token,
        height=grid_height * pixel_per_token,
        grid_width=grid_width,
        grid_height=grid_height,
    )


def resize_native_image(image: Image.Image, plan: NativeImagePlan, divisor: int = 1) -> Image.Image:
    if divisor not in {1, 2, 4, 8}:
        raise ValueError("native image divisor must be 1, 2, 4, or 8")
    if plan.grid_width % divisor or plan.grid_height % divisor:
        raise ValueError("native image grid is not divisible by divisor")
    return image.convert("RGB").resize(
        (plan.width // divisor, plan.height // divisor),
        Image.Resampling.LANCZOS,
    )


class NativeProcessorProxy:
    """Prepare one aligned main input plus the three auxiliary native scales."""

    def __init__(
        self,
        processor: Any,
        *,
        pixel_per_token: int,
        min_pixels: int,
        max_pixels: int,
    ):
        self.processor = processor
        self.pixel_per_token = pixel_per_token
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    def __getattr__(self, name: str) -> Any:
        return getattr(self.processor, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        images = kwargs.get("images")
        videos = kwargs.get("videos")
        if videos is not None:
            raise ValueError("native_multiscale supports images only")
        if images is None:
            return self.processor(*args, **kwargs)
        image_list = list(images) if isinstance(images, (list, tuple)) else [images]
        if not image_list or any(not isinstance(image, Image.Image) for image in image_list):
            raise ValueError("native_multiscale requires PIL image inputs")
        plans = [
            aligned_native_plan(
                image,
                self.pixel_per_token,
                self.min_pixels,
                self.max_pixels,
                grid_multiple=8,
            )
            for image in image_list
        ]
        call_kwargs = dict(kwargs)
        call_kwargs["do_resize"] = False
        call_kwargs["images"] = [
            resize_native_image(image, plan) for image, plan in zip(image_list, plans)
        ]
        main = self.processor(*args, **call_kwargs)
        pending = {}
        for divisor, area_scale in ((2, 4), (4, 16), (8, 64)):
            auxiliary = dict(call_kwargs)
            auxiliary["images"] = [
                resize_native_image(image, plan, divisor)
                for image, plan in zip(image_list, plans)
            ]
            pending[area_scale] = self.processor(*args, **auxiliary)
        main["_tokenfovea_native_inputs"] = pending
        return main
