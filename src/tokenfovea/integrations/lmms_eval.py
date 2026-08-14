from __future__ import annotations

from collections.abc import Callable
import types
from typing import Any

import torch

from lmms_eval.models.simple.qwen2_5_vl import Qwen2_5_VL
from lmms_eval.models.simple.qwen3_5 import Qwen3_5

from tokenfovea.config import FoveaConfig
from tokenfovea.integrations.qwen import install_tokenfovea
from tokenfovea.multiscale import NativeProcessorProxy
from tokenfovea.session import FoveaSession


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _validate_attention_backend(attn_implementation: Any, config: FoveaConfig) -> None:
    if config.mode != "full" and attn_implementation != "sdpa":
        raise ValueError(
            "TokenFovea routed modes require attn_implementation='sdpa', "
            f"got {attn_implementation!r}"
        )


def _config_from_args(
    fovea_budget=None,
    fovea_mode=None,
    fovea_position_mode=None,
    fovea_pooling_mode=None,
    fovea_signal_selection=None,
    fovea_signal_aggregation=None,
    fovea_anchor_window=None,
    fovea_update_interval=None,
    fovea_max_swaps=None,
    fovea_epsilon=None,
    fovea_attention_ema=None,
    fovea_score_mode=None,
    fovea_route_after_prefill=None,
) -> FoveaConfig:
    overrides = {}
    converters: dict[str, tuple[Any, Callable[[Any], Any]]] = {
        "budget": (fovea_budget, int),
        "mode": (fovea_mode, str),
        "position_mode": (fovea_position_mode, str),
        "pooling_mode": (fovea_pooling_mode, str),
        "signal_aggregation": (fovea_signal_aggregation, str),
        "anchor_window": (fovea_anchor_window, float),
        "update_interval": (fovea_update_interval, int),
        "max_swaps": (fovea_max_swaps, int),
        "epsilon": (fovea_epsilon, float),
        "attention_ema": (fovea_attention_ema, float),
        "score_mode": (fovea_score_mode, str),
        "route_after_prefill": (fovea_route_after_prefill, _bool),
    }
    for name, (value, converter) in converters.items():
        if value is not None:
            overrides[name] = converter(value)
    if fovea_signal_selection:
        overrides["signal_selection"] = str(fovea_signal_selection)
    return FoveaConfig(**overrides)


class _TokenFoveaLMMSMixin:
    batch_size: Any
    use_cache: bool
    model: Any
    processor: Any
    device: Any
    min_pixels: Any
    max_pixels: Any

    def _install_fovea(self, config: FoveaConfig) -> None:
        if config.mode != "full" and int(self.batch_size) != 1:
            raise ValueError("TokenFovea requires batch_size=1")
        if config.mode != "full" and not self.use_cache:
            raise ValueError("TokenFovea requires use_cache=True")
        self.fovea_session = FoveaSession(config)
        self.fovea_patch = install_tokenfovea(self.model, self.fovea_session)
        if config.mode != "full" and config.pooling_mode == "native_multiscale":
            vision = self.model.config.vision_config
            patch_size = int(getattr(vision, "patch_size"))
            merge_size = int(getattr(vision, "spatial_merge_size"))
            self.processor = NativeProcessorProxy(
                self.processor,
                pixel_per_token=patch_size * merge_size,
                min_pixels=int(self.min_pixels),
                max_pixels=int(self.max_pixels),
            )
            original_generate = self.model.generate

            def generate_with_native(model, *args, **kwargs):
                pending = kwargs.pop("_tokenfovea_native_inputs", None)
                if pending is None:
                    raise RuntimeError("native_multiscale processor inputs were not prepared")
                device = kwargs.get("input_ids").device if kwargs.get("input_ids") is not None else self.device
                self.fovea_session.begin_native_sample()
                try:
                    for area_scale in (4, 16, 64):
                        auxiliary = pending[area_scale]
                        auxiliary = auxiliary.to(device) if hasattr(auxiliary, "to") else auxiliary
                        self.fovea_session.begin_native_capture(area_scale)
                        with torch.inference_mode():
                            output = model(**auxiliary, use_cache=True, return_dict=True)
                        del output
                        self.fovea_session.end_native_capture()
                    return original_generate(*args, **kwargs)
                except Exception:
                    self.fovea_session.abort_native_sample()
                    raise

            self.model.generate = types.MethodType(generate_with_native, self.model)


class TokenFoveaQwen25VL(_TokenFoveaLMMSMixin, Qwen2_5_VL):
    def __init__(
        self,
        pretrained="Qwen/Qwen2.5-VL-7B-Instruct",
        batch_size=1,
        attn_implementation="sdpa",
        **kwargs,
    ):
        fovea_args = {key: kwargs.pop(key) for key in list(kwargs) if key.startswith("fovea_")}
        fovea_config = _config_from_args(**fovea_args)
        _validate_attention_backend(attn_implementation, fovea_config)
        super().__init__(
            pretrained=pretrained,
            batch_size=batch_size,
            attn_implementation=attn_implementation,
            **kwargs,
        )
        self._install_fovea(fovea_config)


class TokenFoveaQwen35(_TokenFoveaLMMSMixin, Qwen3_5):
    def __init__(
        self,
        pretrained="Qwen/Qwen3.5-9B",
        batch_size=1,
        attn_implementation="sdpa",
        enable_thinking=False,
        **kwargs,
    ):
        fovea_args = {key: kwargs.pop(key) for key in list(kwargs) if key.startswith("fovea_")}
        fovea_config = _config_from_args(**fovea_args)
        _validate_attention_backend(attn_implementation, fovea_config)
        super().__init__(
            pretrained=pretrained,
            batch_size=batch_size,
            attn_implementation=attn_implementation,
            enable_thinking=_bool(enable_thinking),
            **kwargs,
        )
        self._install_fovea(fovea_config)
