import types
import unittest

import torch

from experiments.e3.conditions import get_condition
from experiments.e3.patch import install_e3
from experiments.e3.session import E3Session


class E3PatchTest(unittest.TestCase):
    def test_installer_keeps_qwen35_linear_attention_unmodified(self):
        class Attention(torch.nn.Module):
            def forward(self, hidden_states, **kwargs):
                return hidden_states, None

        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = Attention()

        layers = torch.nn.ModuleList([Layer(), Layer()])
        language = torch.nn.Module()
        language.layers = layers
        language.config = types.SimpleNamespace(
            layer_types=["full_attention", "linear_attention"],
            rope_scaling={"mrope_section": [1]},
        )
        language.rotary_emb = lambda reference, positions: (
            torch.ones_like(reference),
            torch.zeros_like(reference),
        )
        model = torch.nn.Module()
        model.model = types.SimpleNamespace(language_model=language)
        model.config = types.SimpleNamespace(
            model_type="qwen3_5",
            _attn_implementation="sdpa",
            image_token_id=99,
            vision_config=types.SimpleNamespace(spatial_merge_size=2),
        )
        original_linear = layers[1].self_attn.forward
        session = E3Session(get_condition("full_text_anchor"))

        handle = install_e3(model, session)
        try:
            self.assertEqual(session.routed_layers, (0,))
            self.assertEqual(layers[1].self_attn.forward, original_linear)
        finally:
            handle.remove()


if __name__ == "__main__":
    unittest.main()
