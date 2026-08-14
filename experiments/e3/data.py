from __future__ import annotations

from pathlib import Path

from experiments.e2.data import prepare_data as prepare_e2_data

from .config import E3Config


def prepare_data(config: E3Config) -> Path:
    """Validate or prepare the exact E2 sample manifest reused by E3."""
    return prepare_e2_data(config.e2_config())
