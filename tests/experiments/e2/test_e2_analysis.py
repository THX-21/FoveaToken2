import json
import tempfile
import unittest
from pathlib import Path

from experiments.e2.analysis import analyze
from experiments.e2.config import E2Config, ModelSpec
from experiments.e2.report import build_report


class E2AnalysisTest(unittest.TestCase):
    def test_analysis_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = E2Config(
                output_dir=root,
                data_dir=root / "data",
                tasks=("vqav2_val_lite",),
                sample_count=1,
                models={"qwen25": ModelSpec("fake", 1, 2, 28)},
            )
            for condition, score in (
                ("full", 1.0),
                ("lowres_2", 0.0),
                ("uniform2_kv_center", 0.5),
                ("native_uniform4", 1.0),
            ):
                target = root / "qwen25" / condition
                target.mkdir(parents=True)
                (target / "results.json").write_text(json.dumps({
                    "model": "fake", "condition": condition,
                    "tasks": {"vqav2_val_lite": {"primary_score": score, "metrics": {"exact_match": score}, "samples": 1}},
                    "macro_average": score,
                }))
                row = {
                    "sample_id": "vqav2_val_lite:0", "task": "vqav2_val_lite",
                    "metrics": {"exact_match": score}, "generated_token_ids": [1],
                    "token_count_delta": 0,
                    "prefill_seconds": 1.0, "decode_seconds": 2.0, "total_seconds": 3.0,
                }
                (target / "samples.jsonl").write_text(json.dumps(row) + "\n")

            rows = analyze(config, "qwen25")
            report = build_report(config, "qwen25")

            uniform = next(row for row in rows if row["condition"] == "uniform2_kv_center" and row["task"] == "vqav2_val_lite")
            self.assertEqual(uniform["gain_lowres"], 0.5)
            native = next(
                row for row in rows
                if row["condition"] == "native_uniform4" and row["task"] == "vqav2_val_lite"
            )
            self.assertEqual(native["gain_kv_pooling"], 0.5)
            self.assertTrue(report.exists())


if __name__ == "__main__":
    unittest.main()
