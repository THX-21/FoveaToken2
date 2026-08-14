from __future__ import annotations

from typing import Any

import torch

from experiments.e2.patch import E2PatchHandle, install_e2

from .session import E3Session


def install_e3(model: torch.nn.Module, session: E3Session) -> E2PatchHandle:
    """Reuse E2's verified Qwen attention patch with E3 session semantics."""
    return install_e2(model, session)  # type: ignore[arg-type]
