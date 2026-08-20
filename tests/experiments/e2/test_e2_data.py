import json
import tempfile
import unittest
from pathlib import Path

from experiments.e2.config import E2Config, ModelSpec
from experiments.e2.data import validate_manifest
from experiments.e2.evaluator import (
    RANDOM_PERSTEP_PREFILL_PROTOCOL,
    _read_completed_samples,
)


class E2DataManifestTest(unittest.TestCase):
    def test_requires_exact_fixed_subset(self):
        config = E2Config(
            seed=42,
            sample_count=2,
            tasks=("vqav2_val_lite",),
            models={"qwen25": ModelSpec("fake", 1, 2, 28)},
        )
        payload = {
            "seed": 42,
            "tasks": {
                "vqav2_val_lite": [
                    {"sample_id": "vqav2_val_lite:4", "source_index": 4},
                    {"sample_id": "vqav2_val_lite:7", "source_index": 7},
                ]
            },
        }

        validate_manifest(config, payload)
        payload["tasks"]["vqav2_val_lite"].pop()
        with self.assertRaisesRegex(ValueError, "exactly 2"):
            validate_manifest(config, payload)

    def test_reads_completed_samples_for_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            first = {"sample_id": "vqav2_val_lite:4", "metrics": {"exact_match": 1}}
            path.write_text(json.dumps(first) + "\n{", encoding="utf-8")

            completed = _read_completed_samples(path, {"vqav2_val_lite:4", "vqav2_val_lite:7"})

            self.assertEqual(completed, {"vqav2_val_lite:4": first})

    def test_rejects_duplicate_completed_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            row = json.dumps({"sample_id": "vqav2_val_lite:4", "metrics": {}})
            path.write_text(row + "\n" + row + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate"):
                _read_completed_samples(path, {"vqav2_val_lite:4"})

    def test_rejects_obsolete_random_perstep_prefill_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            row = {"sample_id": "vqav2_val_lite:4", "metrics": {}}
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "obsolete E2 random-per-step"):
                _read_completed_samples(
                    path,
                    {"vqav2_val_lite:4"},
                    required_prefill_protocol=RANDOM_PERSTEP_PREFILL_PROTOCOL,
                )


if __name__ == "__main__":
    unittest.main()
