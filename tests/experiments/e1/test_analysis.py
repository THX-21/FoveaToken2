import json
import tempfile
import unittest
from pathlib import Path

from experiments.e1.analysis import analyze, select_with_layer_cap
from experiments.e1.config import E1Config, ModelSpec, NaturalSource
from experiments.e1.report import build_report
from tokenfovea import FoveaConfig, FoveaSession


class E1AnalysisTest(unittest.TestCase):
    def test_layer_cap_is_enforced(self):
        ranked = [
            {"layer": layer, "head": head, "calibrated_gaze_score": 1.0 - index / 100}
            for index, (layer, head) in enumerate([(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (0, 2)])
        ]

        selected = select_with_layer_cap(ranked, requested=4, layer_cap=2)

        self.assertEqual(len(selected), 4)
        self.assertLessEqual(len({row["layer"] for row in selected}), 2)

    def test_analysis_exports_session_compatible_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "qwen25"
            output.mkdir()
            natural = []
            gaze = []
            for layer in range(2):
                for head in range(10):
                    score = 0.3 + (layer * 10 + head) / 1000
                    natural.append(
                        {
                            "layer": layer,
                            "head": head,
                            "samples": 2,
                            "steps": 8,
                            "visual_mass": 0.5,
                            "concentration": score,
                            "coverage": 0.4 + head / 100,
                            "persistence": 0.7 - head / 100,
                        }
                    )
                    gaze.append(
                        {
                            "layer": layer,
                            "head": head,
                            "raw_gaze_score": score,
                            "null_gaze_score": 0.1,
                            "calibrated_gaze_score": score - 0.1,
                            "matrix": [[score if row == column else 0 for column in range(9)] for row in range(9)],
                            "null_vector": [0.1] * 9,
                        }
                    )
            (output / "natural_metrics.json").write_text(json.dumps(natural))
            (output / "gaze_metrics.json").write_text(json.dumps(gaze))
            (output / "probe_metadata.json").write_text(
                json.dumps({"model": "fake", "full_attention_layers": [0, 1], "sample_counts": {}})
            )
            config = E1Config(
                output_dir=root,
                data_dir=root / "data",
                natural_sources=[NaturalSource("x", "x")],
                models={"qwen25": ModelSpec("fake", 1, 2)},
                basic_keep_fraction=1.0,
            )

            analyze(config, "qwen25")
            selection_path = output / "head_selection_top8.json"
            session = FoveaSession(FoveaConfig(signal_selection=str(selection_path)))

            self.assertIsNotNone(session.selected_heads)
            payload = json.loads(selection_path.read_text())
            self.assertEqual(payload["actual_size"], 8)
            self.assertLessEqual(len(session.selected_heads), 4)
            report = build_report(config, "qwen25")
            self.assertTrue(report.exists())
            self.assertTrue((output / "report_assets" / "layer_head_basic_score.png").exists())


if __name__ == "__main__":
    unittest.main()
