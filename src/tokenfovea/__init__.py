"""Fixed-budget multi-scale visual KV routing."""

from .config import FoveaConfig
from .integrations.qwen import install_tokenfovea
from .session import FoveaSession, RouteEvent

__all__ = ["FoveaConfig", "FoveaSession", "RouteEvent", "install_tokenfovea"]
__version__ = "0.1.0"
