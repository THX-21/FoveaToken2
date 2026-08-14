import json
import tempfile
import unittest
from pathlib import Path

from experiments.e2.config import ModelSpec
from experiments.e3.analysis import analyze
from experiments.e3.conditions import CONDITIONS
from experiments.e3.config import E3Config
from experiments.e3.evaluator import PROMPT_VERSION


class E3AnalysisTest(unittest.TestCase):
    def _fixture(self, root: Path) -> E3Config:
        config = E3Config(
            sample_count=1,
            tasks=("vqav2_val_lite",),
            output_dir=root,
            models={"qwen25": ModelSpec("test/model", 16, 64, 4)},
        )
        for condition in CONDITIONS:
            directory = root / "qwen25" / condition.name
            directory.mkdir(parents=True)
            score = 0.5 if condition.text_anchor else 1.0
            (directory / "results.json").write_text(
                json.dumps(
                    {
                        "condition": condition.name,
                        "prompt_version": PROMPT_VERSION,
                        "tasks": {
                            "vqav2_val_lite": {
                                "primary_score": score,
                                "samples": 1,
                            }
                        },
                        "macro_average": score,
                    }
                ),
                encoding="utf-8",
            )
            row = {
                "sample_id": "vqav2_val_lite:0",
                "task": "vqav2_val_lite",
                "prediction": "answer",
                "raw_prediction": "Analyze: concise evidence\nAnswer: answer",
                "analyze": "concise evidence",
                "format_compliant": True,
                "analyze_word_count": 2,
                "analyze_within_limit": True,
                "metrics": {"exact_match": int(not condition.text_anchor)},
                "generated_token_ids": [7, 8 if condition.text_anchor else 9],
                "prefill_seconds": 1.0,
                "decode_seconds": 1.0,
                "native_prefill_seconds": 0.0,
                "total_seconds": 2.0,
                "visual_tokens": 16,
                "active_tokens": 4 if condition.compact else 16,
                "compression_ratio": 0.25 if condition.compact else 1.0,
            }
            if condition.text_anchor:
                row.update({"anchor_position_min": 1.0, "anchor_position_max": 2.0})
            (directory / "samples.jsonl").write_text(
                json.dumps(row) + "\n", encoding="utf-8"
            )
        return config

    def test_analysis_requires_and_reports_first_token_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._fixture(Path(directory))
            rows = analyze(config, "qwen25")
            row = next(
                row
                for row in rows
                if row["condition"] == "pool2_text_anchor"
                and row["task"] == "vqav2_val_lite"
            )
            self.assertEqual(row["first_token_agreement"], 1.0)
            self.assertEqual(row["delta_control"], -0.5)
            self.assertEqual(len(rows), 16)

    def test_analysis_rejects_prefill_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._fixture(Path(directory))
            path = config.output_dir / "qwen25" / "full_text_anchor" / "samples.jsonl"
            row = json.loads(path.read_text(encoding="utf-8"))
            row["generated_token_ids"][0] = 999
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prefill mismatch"):
                analyze(config, "qwen25")


if __name__ == "__main__":
    unittest.main()
