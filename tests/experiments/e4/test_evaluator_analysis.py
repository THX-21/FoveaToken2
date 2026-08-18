import torch

from experiments.e4.analysis import _pair_metrics, _roi_fine_gain
from experiments.e4.evaluator import parse_reasoning_response
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
