from __future__ import annotations

import torch

from tokenfovea.generation import (
    generate_with_prefill_boundary,
    prompt_prefix_inputs,
)


def test_prompt_prefix_inputs_trims_only_token_aligned_values():
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "position_ids": torch.arange(3).repeat(3, 1, 1),
        "cache_position": torch.arange(3),
        "pixel_values": torch.randn(4, 8),
    }

    prefix = prompt_prefix_inputs(inputs)

    assert prefix["input_ids"].tolist() == [[1, 2]]
    assert prefix["attention_mask"].shape[-1] == 2
    assert prefix["position_ids"].shape[-1] == 2
    assert prefix["cache_position"].shape[-1] == 2
    assert prefix["pixel_values"] is inputs["pixel_values"]


def test_generate_with_prefill_boundary_passes_prefix_cache_to_generate():
    class Output:
        past_key_values = object()

    class Model:
        def __init__(self):
            self.prefill = None
            self.generation = None

        def __call__(self, **kwargs):
            self.prefill = kwargs
            return Output()

        def generate(self, **kwargs):
            self.generation = kwargs
            return "generated"

    model = Model()
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }

    result = generate_with_prefill_boundary(model, inputs, {"max_new_tokens": 2})

    assert result == "generated"
    assert model.prefill["input_ids"].tolist() == [[1, 2]]
    assert model.generation["input_ids"].tolist() == [[1, 2, 3]]
    assert model.generation["past_key_values"] is Output.past_key_values
