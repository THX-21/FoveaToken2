"""Fixed-budget multi-scale visual KV routing."""

from .config import FoveaConfig
from .generation import generate_with_prefill_boundary, prompt_prefix_inputs
from .integrations.qwen import install_tokenfovea
from .session import FoveaSession, RouteEvent

__all__ = [
    "FoveaConfig",
    "FoveaSession",
    "RouteEvent",
    "generate_with_prefill_boundary",
    "install_tokenfovea",
    "prompt_prefix_inputs",
]
__version__ = "0.1.0"
