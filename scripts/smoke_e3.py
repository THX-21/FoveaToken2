from __future__ import annotations

import torch

from experiments.e3.conditions import get_condition
from experiments.e3.session import E3Session


def _encoder(reference: torch.Tensor, positions: torch.Tensor):
    shape = (reference.shape[0], positions.shape[-1], reference.shape[-1])
    return torch.ones(shape), torch.zeros(shape)


def main() -> None:
    session = E3Session(get_condition("pool2_text_anchor"), anchor_window=8)
    session.attach([0], _encoder, "qwen2_5_vl")
    session.rotate_key = lambda key, cos, sin: key
    session.begin_sample("smoke")
    ids = torch.tensor([[1, *([99] * 16), 2]])
    session.configure_prompt("smoke", ids, torch.tensor([[1, 8, 8]]), 99, 2)
    session.observe_position_ids(torch.arange(18).view(1, 1, -1).expand(3, -1, -1))
    raw = torch.arange(18, dtype=torch.float32).view(1, 1, 18, 1)
    session.capture_layer(0, raw, raw + 100, raw, torch.zeros(1, 18, 1), None)
    prefill = session.prefill_compact(0, raw, raw, raw)
    assert prefill is not None and prefill[0].shape[-2] == 6
    session.observe_position_ids(torch.full((3, 1, 1), 18))
    full = torch.arange(19, dtype=torch.float32).view(1, 1, 19, 1)
    keys, values, mask = session.decode_compact(0, full, full, full[..., -1:, :])
    assert keys.shape == values.shape and keys.shape[-2] == 7
    assert mask.shape == (1, 1, 1, 7)
    assert session.anchor_min is not None and session.anchor_max is not None
    print("E3 CPU smoke test passed")


if __name__ == "__main__":
    main()
