from __future__ import annotations

from collections.abc import Callable

from tokenfovea.config import FoveaConfig
from tokenfovea.session import FoveaSession, RouteEvent


class E4Session(FoveaSession):
    """Core session with decode signal collection disabled for prefill-static E4."""

    def __init__(
        self,
        config: FoveaConfig,
        *,
        prefill_static: bool = False,
        route_observer: Callable[[RouteEvent], None] | None = None,
    ):
        self.prefill_static = prefill_static
        super().__init__(config, route_observer=route_observer)

    def needs_signal(self, layer_idx: int) -> bool:
        if self.prefill_static and layer_idx in self.pyramids:
            return False
        return super().needs_signal(layer_idx)
