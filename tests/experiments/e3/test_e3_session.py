import unittest

import torch

from experiments.e3.conditions import get_condition
from experiments.e3.session import E3Session


def _encoder(reference, positions):
    cos = positions[1].unsqueeze(1).unsqueeze(-1)
    return cos, torch.zeros_like(cos)


def _session(condition_name: str) -> tuple[E3Session, torch.Tensor]:
    session = E3Session(get_condition(condition_name), anchor_window=8)
    session.attach([0], _encoder, "qwen2_5_vl")
    session.rotate_key = lambda key, cos, sin: key + cos
    session.begin_sample("sample")
    ids = torch.tensor([[1, *([99] * 16), 2]])
    session.configure_prompt("sample", ids, torch.tensor([[1, 8, 8]]), 99, 2)
    session.observe_position_ids(torch.arange(18).view(1, 1, -1).expand(3, -1, -1))
    raw = torch.arange(18, dtype=torch.float32).view(1, 1, 18, 1)
    session.capture_layer(0, raw, raw + 100, raw + 200, torch.zeros(1, 18, 1), None)
    return session, raw


class E3SessionTest(unittest.TestCase):
    def test_pool2_control_uses_center_positions_during_decode(self):
        session, raw = _session("pool2_center")
        self.assertTrue(session.enabled)
        prefill = session.prefill_compact(0, raw, raw, raw)
        self.assertIsNotNone(prefill)
        session.observe_position_ids(torch.full((3, 1, 1), 20))
        full = torch.arange(19, dtype=torch.float32).view(1, 1, 19, 1)
        keys, _, _ = session.decode_compact(0, full, full, full[..., -1:, :])
        self.assertAlmostEqual(float(keys[0, 0, 0, 0]), 7.0)

    def test_full_keeps_prefill_and_replaces_only_visual_decode_keys(self):
        session, raw = _session("full_text_anchor")
        self.assertTrue(session.preserve_prefill)
        self.assertIsNone(session.prefill_compact(0, raw, raw, raw))
        session.observe_position_ids(torch.full((3, 1, 1), 20))
        full = torch.arange(19, dtype=torch.float32).view(1, 1, 19, 1)
        values = full + 1000
        keys, output_values, mask = session.decode_compact(0, full, values, full[..., -1:, :])

        self.assertTrue(torch.equal(output_values, values))
        self.assertEqual(mask.shape, (1, 1, 1, 19))
        self.assertEqual(float(keys[0, 0, 0, 0]), float(full[0, 0, 0, 0]))
        self.assertAlmostEqual(float(keys[0, 0, 1, 0]), 1 + 20 - 8 + 9 * 0.125)
        self.assertEqual(float(keys[0, 0, -1, 0]), float(full[0, 0, -1, 0]))

    def test_pool2_prefill_uses_center_and_decode_uses_same_pooled_values(self):
        session, raw = _session("pool2_text_anchor")
        self.assertFalse(session.preserve_prefill)
        prefill = session.prefill_compact(0, raw, raw + 1000, raw)
        assert prefill is not None
        self.assertEqual(prefill[0].shape[-2], 6)
        session.observe_position_ids(torch.full((3, 1, 1), 20))
        full = torch.arange(19, dtype=torch.float32).view(1, 1, 19, 1)
        keys, values, mask = session.decode_compact(0, full, full + 1000, full[..., -1:, :])

        self.assertEqual(keys.shape[-2], 7)
        self.assertEqual(mask.shape, (1, 1, 1, 7))
        self.assertAlmostEqual(float(values[0, 0, 0, 0]), 103.5)
        self.assertAlmostEqual(float(keys[0, 0, 0, 0]), 3.5 + 20 - 8 + 9 * 0.25)

    def test_native2_reads_only_scale4_bank(self):
        session = E3Session(get_condition("native2_text_anchor"), anchor_window=8)
        session.attach([0], _encoder, "qwen2_5_vl")
        session.rotate_key = lambda key, cos, sin: key
        session.begin_sample("native")
        session.begin_native_capture(4)
        auxiliary_ids = torch.tensor([[1, *([99] * 4), 2]])
        session.configure_native_capture_prompt(
            auxiliary_ids, torch.tensor([[1, 4, 4]]), 99, 2
        )
        auxiliary = torch.tensor([[[[0.0], [10.0], [20.0], [30.0], [0.0], [0.0]]]])
        session.capture_layer(
            0,
            auxiliary,
            auxiliary + 100,
            auxiliary,
            torch.zeros(1, 6, 1),
            None,
        )
        self.assertIsNone(
            session.prefill_compact(0, auxiliary, auxiliary, auxiliary)
        )
        session.end_native_capture()
        ids = torch.tensor([[1, *([99] * 16), 2]])
        session.configure_prompt("native", ids, torch.tensor([[1, 8, 8]]), 99, 2)
        session.observe_position_ids(torch.arange(18).view(1, 1, -1).expand(3, -1, -1))
        main = torch.ones(1, 1, 18, 1)
        session.capture_layer(0, main, main, main, torch.zeros(1, 18, 1), None)
        prefill = session.prefill_compact(0, main, main, main)
        assert prefill is not None

        self.assertEqual(prefill[0][0, 0, :4, 0].tolist(), [10.0, 20.0, 30.0, 0.0])
        self.assertEqual(session.native_bank_tokens, 20)


if __name__ == "__main__":
    unittest.main()
