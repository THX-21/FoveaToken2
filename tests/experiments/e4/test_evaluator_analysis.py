import json

import pytest
import torch

from experiments.e4.analysis import (
    _baseline,
    _pair_metrics,
    _roi_fine_gain,
    _structured_correct,
)
from experiments.e4.conditions import Condition
from experiments.e4.evaluator import (
    NATIVE_PREFILL_PROTOCOL,
    _chat_completions_url,
    _generate_with_routed_prompt,
    _metric_correct,
    _prompt_prefix_inputs,
    _read_samples,
    _suite_prompt,
    parse_reasoning_response,
)
from experiments.e4.runtime import RouteTraceObserver
from experiments.e4.session import E4Session
from tokenfovea.config import FoveaConfig
from tokenfovea.router import SplitMergeRouter
from tokenfovea.session import FoveaSession
from tokenfovea.session import RouteEvent
from tokenfovea.topology import DeviceTreeTopology, VisualTokenForest


def test_reasoning_parser_prefers_answer_tag():
    answer, analysis, compliant = parse_reasoning_response(
        "<analysis>The target is red.</analysis>\n<answer>red</answer>"
    )
    assert (answer, analysis, compliant) == ("red", "The target is red.", True)


def test_compression_uses_formal_prompt_while_reasoning_appends_instruction():
    assert _suite_prompt("question", "formal") == "question"
    assert _suite_prompt("question", "compression") == "question"
    assert _suite_prompt("question", "reasoning").startswith("question\n\nAnalyze")


def test_hrbench_judge_accepts_base_or_complete_api_url():
    assert _chat_completions_url("https://judge.example/v1") == (
        "https://judge.example/v1/chat/completions"
    )
    assert _chat_completions_url("https://judge.example/v1/chat/completions/") == (
        "https://judge.example/v1/chat/completions"
    )


def test_route_observer_serializes_scales_without_default_core_sync():
    forest = VisualTokenForest.from_aligned_grids([(8, 8)])
    leaf = next(node.node_id for node in forest.nodes if node.valid_count == 1)
    root = forest.roots[0]
    observer = RouteTraceObserver()
    observer.begin_sample("sample")
    observer(
        RouteEvent(
            phase="prefill",
            step=0,
            updated=True,
            active_before=torch.tensor([root]),
            active_after=torch.tensor([leaf]),
            node_scores=torch.arange(len(forest.nodes), dtype=torch.float32),
            swaps=torch.tensor(1),
            forest=forest,
        )
    )
    row = observer.drain()[0]
    assert row["active_before"][0]["area_scale"] == 64
    assert row["active_after"][0]["area_scale"] == 1
    assert row["swaps"] == 1


def test_session_emits_prefill_route_event_only_when_observed():
    events = []
    session = FoveaSession(
        FoveaConfig(mode="dynamic", budget=4), route_observer=events.append
    )
    forest = VisualTokenForest.from_grids([(2, 2)])
    topology = DeviceTreeTopology.build(forest, torch.device("cpu"))
    session.forest = forest
    session.topology = topology
    session._routing_device = torch.device("cpu")
    initial = torch.tensor(sorted(forest.initial_front(4)))
    session.router = SplitMergeRouter(topology, initial)
    session.prompt_signal_sum = torch.ones(4)
    session.prompt_signal_count = 1
    session._finish_prefill()
    assert len(events) == 1
    assert events[0].phase == "prefill"
    assert events[0].active_before.numel() == 4


def test_prefill_static_session_does_not_collect_decode_signals():
    session = E4Session(FoveaConfig(mode="dynamic"), prefill_static=True)
    session.pyramids[3] = object()  # type: ignore[assignment]
    assert not session.needs_signal(3)
    assert session.needs_signal(4)


def test_pair_metrics_and_roi_diagnostic():
    baseline = {
        "x": {"task": "t", "correct": True, "generated_token_ids": [1, 2, 3]}
    }
    current = {
        "x": {"task": "t", "correct": True, "generated_token_ids": [1, 4, 3]}
    }
    metrics = _pair_metrics(baseline, current, "t")
    assert metrics["first_token_agreement"] == 1.0
    assert metrics["mean_first_divergence"] == 1.0
    row = {
        "roi": [0.0, 0.0, 0.5, 0.5],
        "original_size": [100, 100],
        "highres_grid": [8, 8],
        "final_front": [
            {"x0": 0, "x1": 1, "y0": 0, "y1": 1, "area_scale": 1},
            {"x0": 4, "x1": 8, "y0": 4, "y1": 8, "area_scale": 16},
        ],
    }
    assert _roi_fine_gain(row) > 0


def test_pair_metrics_reports_first_token_mismatch_as_diagnostic():
    baseline = {
        "x": {"task": "t", "correct": True, "generated_token_ids": [1, 2]}
    }
    current = {
        "x": {"task": "t", "correct": False, "generated_token_ids": [3, 2]}
    }
    metrics = _pair_metrics(baseline, current, "t")
    assert metrics["first_token_agreement"] == 0.0
    assert metrics["mean_first_divergence"] == 0.0


def test_compression_conditions_use_matching_lowres_and_formal_full_baselines():
    lowres = Condition("lowres6", "lowres", 6.0)
    dynamic = Condition("dynamic6_top8_native", "dynamic", 6.0, True)
    assert _baseline("compression", lowres) == ("formal", "full")
    assert _baseline("compression", dynamic) == ("compression", "lowres6")


def test_structured_multiple_choice_correctness():
    metrics = {
        "xlrs_micro_score": {"pred_answer": "BD", "answer": "DB"},
    }
    assert _metric_correct(metrics) is True
    assert _structured_correct(metrics) is True


def test_native_resume_rejects_obsolete_prefill_samples(tmp_path):
    path = tmp_path / "samples.jsonl"
    path.write_text(json.dumps({"sample_id": "task:0"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="obsolete Native prefill protocol"):
        _read_samples(
            path,
            {"task:0"},
            native_prefill_protocol=NATIVE_PREFILL_PROTOCOL,
        )


def test_resume_rejects_old_spatial_divisor_condition_names(tmp_path):
    path = tmp_path / "samples.jsonl"
    path.write_text(
        json.dumps(
            {
                "sample_id": "task:0",
                "configured_compression_ratio": 4,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="obsolete compression-ratio protocol"):
        _read_samples(path, {"task:0"}, compression_ratio=2)


def test_routed_prompt_prefix_removes_only_the_final_token():
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "position_ids": torch.arange(3).repeat(3, 1, 1),
        "pixel_values": torch.randn(4, 8),
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
    }
    prefix = _prompt_prefix_inputs(inputs)
    assert prefix["input_ids"].tolist() == [[1, 2]]
    assert prefix["attention_mask"].shape[-1] == 2
    assert prefix["position_ids"].shape[-1] == 2
    assert prefix["pixel_values"] is inputs["pixel_values"]
    assert prefix["image_grid_thw"] is inputs["image_grid_thw"]


def test_routed_generation_prefills_prefix_then_passes_cache_to_full_prompt():
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
    result = _generate_with_routed_prompt(model, inputs, {"max_new_tokens": 2})
    assert result == "generated"
    assert model.prefill["input_ids"].tolist() == [[1, 2]]
    assert model.generation["input_ids"].tolist() == [[1, 2, 3]]
    assert model.generation["past_key_values"] is Output.past_key_values
