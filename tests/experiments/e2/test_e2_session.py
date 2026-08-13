import tempfile
import unittest
from pathlib import Path

import torch

from experiments.e2.conditions import get_condition
from experiments.e2.front import BlockFront
from experiments.e2.session import LayerSource
from experiments.e2.session import E2Session


def _encoder(reference, positions):
    shape = (reference.shape[0], positions.shape[-1], reference.shape[-1])
    return torch.ones(shape), torch.zeros(shape)


class E2SessionTest(unittest.TestCase):
    def _session(self, condition_name):
        session = E2Session(get_condition(condition_name), seed=42)
        session.attach([0, 4], _encoder, "qwen2_5_vl")
        session.begin_sample("sample")
        input_ids = torch.tensor([[1, *([99] * 64), 2, 3]])
        session.configure_prompt("sample", input_ids, torch.tensor([[1, 16, 16]]), 99, 2)
        positions = torch.arange(input_ids.shape[-1]).view(1, -1).expand(3, -1).unsqueeze(1)
        session.observe_position_ids(positions)
        return session

    def test_fixed_front_is_shared_by_prefill_and_decode(self):
        session = self._session("random_fixed_kv_center")
        front_hash = session.fixed_front.digest()

        self.assertEqual(session.fixed_front.digest(), front_hash)
        self.assertEqual(session.decode_step, 0)

    def test_perstep_front_shared_across_layers_then_changes(self):
        session = self._session("random_perstep_kv_center")
        raw = torch.arange(67, dtype=torch.float32).view(1, 1, 67, 1)
        values = raw + 1
        rotated = raw.clone()
        hidden = torch.zeros(1, 67, 1)
        for layer in (0, 4):
            session.capture_layer(layer, raw, values, rotated, hidden, None)
        full_keys = torch.arange(67, dtype=torch.float32).view(1, 1, 67, 1)
        first = session.decode_compact(0, full_keys, full_keys, raw)
        first_hash = session.step_front.digest()
        session.finish_layer(0)
        session.decode_compact(4, full_keys, full_keys, raw)
        self.assertEqual(session.step_front.digest(), first_hash)
        session.finish_layer(4)
        self.assertIsNone(session.step_front)
        session.decode_compact(0, full_keys, full_keys, raw)
        self.assertNotEqual(session.step_front.digest(), first_hash)
        self.assertEqual(first[0].shape[-2], session.step_front.node_count + len(session.text_positions))

    def test_prefill_mask_only_targets_post_image_text(self):
        session = self._session("uniform2_kv_center")
        raw = torch.arange(67, dtype=torch.float32).view(1, 1, 67, 1)
        session.capture_layer(0, raw, raw, raw, torch.zeros(1, 67, 1), None)
        full = torch.arange(67, dtype=torch.float32).view(1, 1, 67, 1)

        _, _, queries, mask = session.prefill_compact(0, full, full, raw)

        self.assertEqual(queries.tolist(), [65, 66])
        self.assertEqual(mask.shape[-2], 2)
        self.assertFalse(bool(mask[0, 0, 0, -1]))
        self.assertTrue(bool(mask[0, 0, 1].all()))

    def test_kv_hidden_and_postrope_pooling_semantics(self):
        front = BlockFront.uniform(4, 4, 4)
        raw = torch.arange(16, dtype=torch.float32).view(1, 1, 16, 1)
        values = raw + 10
        rotated = raw + 100
        hidden = torch.arange(32, dtype=torch.float32).view(1, 16, 2)
        positions = torch.arange(16, dtype=torch.float32).view(1, 1, 16).expand(3, -1, -1)
        source = LayerSource(
            raw, values, rotated, hidden, positions,
            lambda nodes: (nodes[..., :1].unsqueeze(1) ** 2, (nodes[..., 1:].unsqueeze(1) + 1)),
        )
        rotate = lambda key, cos, sin: key

        kv_keys, kv_values = source.gather(
            front, get_condition("uniform4_kv_center"), _encoder, raw, rotate
        )
        hidden_keys, hidden_values = source.gather(
            front, get_condition("uniform4_hidden_center"), _encoder, raw, rotate
        )
        post_keys, post_values = source.gather(
            front, get_condition("uniform4_postrope"), _encoder, raw, rotate
        )

        self.assertAlmostEqual(float(kv_keys[0, 0, 0, 0]), 7.5)
        self.assertAlmostEqual(float(kv_values[0, 0, 0, 0]), 17.5)
        self.assertAlmostEqual(float(hidden_keys[0, 0, 0, 0]), float(hidden[..., 0].mean() ** 2))
        self.assertAlmostEqual(float(hidden_values[0, 0, 0, 0]), float(hidden[..., 1].mean() + 1))
        self.assertAlmostEqual(float(post_keys[0, 0, 0, 0]), 107.5)
        self.assertAlmostEqual(float(post_values[0, 0, 0, 0]), 17.5)


if __name__ == "__main__":
    unittest.main()
