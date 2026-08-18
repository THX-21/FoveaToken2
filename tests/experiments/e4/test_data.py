from pathlib import Path

from experiments.e4.config import E4Config, ModelSpec
from experiments.e4.data import prepare_data, suite_indices


class FakeTask:
    def __init__(self, docs):
        self.eval_docs = docs


def _config(tmp_path: Path) -> E4Config:
    tasks = (
        "vstar_bench",
        "visualprobe_easy",
        "visualprobe_medium",
        "visualprobe_hard",
        "finers_qa",
        "hrscene_testmini",
    )
    return E4Config(
        mechanism_count=4,
        data_dir=tmp_path,
        formal_tasks=tasks,
        mechanism_tasks=tasks,
        primary_metrics={name: "score" for name in tasks},
        head_selections={"qwen25": tmp_path / "heads.json"},
        models={"qwen25": ModelSpec("model", 64, 4096 * 28**2, 28)},
    )


def test_manifest_filters_finers_and_stratifies_visualprobe(tmp_path):
    config = _config(tmp_path)
    tasks = {
        "vstar_bench": FakeTask([{"category": f"c{i % 2}"} for i in range(8)]),
        "visualprobe_easy": FakeTask([{} for _ in range(5)]),
        "visualprobe_medium": FakeTask([{} for _ in range(6)]),
        "visualprobe_hard": FakeTask([{} for _ in range(7)]),
        "finers_qa": FakeTask(
            [
                {"annotations": {"A": "" if i < 2 else "a", "Q-type": f"q{i % 2}"}}
                for i in range(10)
            ]
        ),
        "hrscene_testmini": FakeTask([{"data_source": f"s{i % 2}"} for i in range(8)]),
    }
    path = prepare_data(config, tasks)
    assert path.is_file()
    formal = suite_indices(config, "formal")
    assert formal["finers_qa"] == list(range(2, 10))
    reasoning = suite_indices(config, "reasoning")
    assert sum(len(reasoning[name]) for name in reasoning if name.startswith("visualprobe")) == 4
    assert len(reasoning["vstar_bench"]) == 4
    assert len(reasoning["finers_qa"]) == 4
    assert len(reasoning["hrscene_testmini"]) == 4
