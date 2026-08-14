import json
import tempfile
import unittest
from pathlib import Path

import torch

from tokenfovea.config import FoveaConfig
from tokenfovea.session import FoveaSession


class SessionTest(unittest.TestCase):
    def test_native_multiscale_auxiliary_lifecycle_and_scale64_root(self):
        session = FoveaSession(
            FoveaConfig(mode="uniform", budget=1, pooling_mode="native_multiscale")
        )
        session.attach(
            [0],
            lambda reference, positions: (positions, positions),
            lambda key, cos, sin: key,
        )
        session.begin_native_sample()
        for area, llm_grid in ((4, 4), (16, 2), (64, 1)):
            session.begin_native_capture(area)
            input_ids = torch.tensor([[1] + [99] * (llm_grid**2) + [2]])
            session.configure_native_capture_prompt(
                input_ids,
                torch.tensor([[1, llm_grid * 2, llm_grid * 2]]),
                image_token_id=99,
                spatial_merge_size=2,
            )
            raw = torch.full((1, 1, input_ids.shape[-1], 1), float(area))
            session.capture_prefill_layer(0, raw, raw + 100, raw, None)
            session.end_native_capture()

        main_ids = torch.tensor([[1] + [99] * 64 + [2]])
        session.configure_prompt(
            main_ids,
            torch.tensor([[1, 16, 16]]),
            image_token_id=99,
            spatial_merge_size=2,
        )
        positions = torch.arange(main_ids.shape[-1]).view(1, 1, -1).expand(3, -1, -1)
        session.observe_position_ids(positions)
        raw = torch.ones(1, 1, main_ids.shape[-1], 1)
        session.capture_prefill_layer(0, raw, raw + 100, raw, None)

        full = torch.zeros(1, 1, main_ids.shape[-1] + 1, 1)
        keys, values, active = session.compose(0, full, full, raw[..., :1, :])

        self.assertEqual(active.numel(), 1)
        self.assertEqual(float(keys[0, 0, 0, 0]), 64.0)
        self.assertEqual(float(values[0, 0, 0, 0]), 164.0)
        self.assertTrue(all(not layers for layers in session.native_sources.values()))

    def test_signal_selection_keeps_only_requested_heads(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "heads.json"
            path.write_text(
                json.dumps({"selected_heads": [{"layer": 0, "head": 1}]}),
                encoding="utf-8",
            )
            session = FoveaSession(FoveaConfig(signal_selection=str(path)))
            signal = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
            reduced, count = session._reduce_signal(0, signal)
            self.assertEqual(count, 1)
            self.assertTrue(torch.equal(reduced, signal[1]))
            self.assertIsNone(session._reduce_signal(1, signal))

    def test_reset_prompt_discards_previous_visual_state(self):
        session = FoveaSession(FoveaConfig())
        session.configure_prompt(
            torch.tensor([[1] + [99] * 4 + [2]]),
            torch.tensor([[1, 4, 4]]),
            image_token_id=99,
            spatial_merge_size=2,
        )
        self.assertTrue(session.is_configured)
        session.reset_prompt()
        self.assertFalse(session.is_configured)
        self.assertEqual(session.visual_positions, [])
        self.assertEqual(session.pyramids, {})

    def test_uniform_text_anchor_refreshes_every_decode_step(self):
        session = FoveaSession(FoveaConfig(mode="uniform", budget=4, position_mode="text_anchor"))
        session.attach(
            [0],
            lambda reference, positions: (positions, positions),
            lambda key, cos, sin: key,
        )
        session.configure_prompt(
            torch.tensor([[1] + [99] * 16 + [2]]),
            torch.tensor([[1, 8, 8]]),
            image_token_id=99,
            spatial_merge_size=2,
        )
        session.observe_position_ids(torch.arange(18).repeat(3, 1, 1))
        prompt_keys = torch.randn(1, 1, 18, 4)
        session.capture_prefill_layer(0, prompt_keys, prompt_keys, prompt_keys, None)

        session.observe_position_ids(torch.full((3, 1, 1), 18))
        full_keys = torch.randn(1, 1, 19, 4)
        _, _, active_ids = session.compose(0, full_keys, full_keys, full_keys[..., -1:, :])
        first_anchor = session._anchor_positions(active_ids, torch.device("cpu")).clone()
        session.record_decode_layer(0, None)
        self.assertEqual(session._decode_text_indices, {})

        session.observe_position_ids(torch.full((3, 1, 1), 19))
        full_keys = torch.randn(1, 1, 20, 4)
        _, _, active_ids = session.compose(0, full_keys, full_keys, full_keys[..., -1:, :])
        second_anchor = session._anchor_positions(active_ids, torch.device("cpu"))
        self.assertFalse(torch.equal(first_anchor, second_anchor))

    def test_decode_batch_expansion_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "beam search"):
            FoveaSession.validate_decode_batch(2)

    def test_keeps_a_fixed_device_front(self):
        for position_mode in ("native_center", "text_anchor", "no_rope", "post_rope_pool"):
            with self.subTest(position_mode=position_mode):
                config = FoveaConfig(budget=7, max_swaps=1, position_mode=position_mode)
                session = FoveaSession(config)
                session.attach(
                    [0],
                    lambda reference, positions: (positions, positions),
                    lambda key, cos, sin: key,
                )
                input_ids = torch.tensor([[1] + [99] * 16 + [2]])
                session.configure_prompt(
                    input_ids,
                    torch.tensor([[1, 8, 8]]),
                    image_token_id=99,
                    spatial_merge_size=2,
                )
                session.observe_position_ids(torch.arange(18).repeat(3, 1, 1))

                raw_keys = torch.randn(1, 2, 18, 4)
                values = torch.randn_like(raw_keys)
                session.capture_prefill_layer(0, raw_keys, values, raw_keys, torch.rand(16))
                full_keys = torch.randn(1, 2, 19, 4)
                full_values = torch.randn_like(full_keys)
                keys, composed_values, active_ids = session.compose(
                    0,
                    full_keys,
                    full_values,
                    raw_keys[..., -1:, :],
                )

                self.assertEqual(keys.shape[-2], active_ids.numel() + 3)
                self.assertEqual(composed_values.shape, keys.shape)
                session.record_decode_layer(
                    0,
                    torch.full((active_ids.numel(),), 1.0 / active_ids.numel()),
                )
                self.assertEqual(session.router.active_ids().numel(), active_ids.numel())
