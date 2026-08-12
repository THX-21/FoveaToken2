"""E1: training-free discovery of visual routing attention heads."""

from .metrics import gaze_statistics, hybrid_statistics
from .probe import E1AttentionProbe, full_context_attention

__all__ = ["E1AttentionProbe", "full_context_attention", "gaze_statistics", "hybrid_statistics"]
