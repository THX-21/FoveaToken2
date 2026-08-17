import json
import tempfile
import unittest
from pathlib import Path

from experiments.distributed import (
    DistributedContext,
    merge_rank_jsonl,
    merge_rank_text,
)
from experiments.e1.probe import merge_probe_checkpoints


class DistributedUtilitiesTest(unittest.TestCase):
    def test_shards_samples_by_rank(self):
        values = list(range(10))
        shards = [DistributedContext(rank, rank, 3).shard(values) for rank in range(3)]

        self.assertEqual(shards, [[0, 3, 6, 9], [1, 4, 7], [2, 5, 8]])
        self.assertEqual(sorted(value for shard in shards for value in shard), values)

    def test_rank_path_preserves_suffix(self):
        context = DistributedContext(rank=2, local_rank=2, world_size=4)
        self.assertEqual(
            context.rank_path(Path("output/samples.jsonl")),
            Path("output/samples.rank2.jsonl"),
        )

    def test_merges_rank_jsonl_and_recovers_truncated_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "samples.jsonl"
            destination.write_text('{"sample_id":"old","value":0}\n', encoding="utf-8")
            (root / "samples.rank0.jsonl").write_text(
                '{"sample_id":"a","value":1}\n', encoding="utf-8"
            )
            (root / "samples.rank1.jsonl").write_text(
                '{"sample_id":"b","value":2}\n{"sample_id":', encoding="utf-8"
            )

            count = merge_rank_jsonl(destination, key="sample_id")
            rows = [json.loads(line) for line in destination.read_text().splitlines()]

            self.assertEqual(count, 3)
            self.assertEqual({row["sample_id"] for row in rows}, {"old", "a", "b"})
            self.assertFalse((root / "samples.rank0.jsonl").exists())
            self.assertFalse((root / "samples.rank1.jsonl").exists())

    def test_merges_rank_text_in_rank_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "trace.jsonl"
            destination.write_text("old\n", encoding="utf-8")
            (root / "trace.rank1.jsonl").write_text("one\n", encoding="utf-8")
            (root / "trace.rank0.jsonl").write_text("zero\n", encoding="utf-8")

            merge_rank_text(destination)

            self.assertEqual(destination.read_text(encoding="utf-8"), "old\nzero\none\n")

    def test_merges_e1_accumulators(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [root / "rank0.json", root / "rank1.json"]
            for rank, path in enumerate(paths):
                payload = {
                    "completed_sample_ids": [f"sample-{rank}"],
                    "natural_totals": {
                        "0:0": {
                            "samples": 1,
                            "steps": rank + 1,
                            "visual_mass": 0.25 + rank,
                            "concentration": 0.5,
                            "coverage": 0.1,
                            "persistence": 0.2,
                        }
                    },
                    "gaze_totals": {
                        "0:0": {
                            "matrix": [[float(rank)] * 9 for _ in range(9)],
                            "matrix_count": [1] * 9,
                            "null": [float(rank)] * 9,
                            "null_count": 1,
                        }
                    },
                    "sample_counts": {"natural": 1, "gaze": rank, "null": 1 - rank},
                }
                path.write_text(json.dumps(payload), encoding="utf-8")
            destination = root / "merged.json"

            count = merge_probe_checkpoints(paths, destination)
            merged = json.loads(destination.read_text(encoding="utf-8"))

            self.assertEqual(count, 2)
            self.assertEqual(merged["completed_sample_ids"], ["sample-0", "sample-1"])
            self.assertEqual(merged["natural_totals"]["0:0"]["samples"], 2.0)
            self.assertEqual(merged["natural_totals"]["0:0"]["steps"], 3.0)
            self.assertEqual(merged["gaze_totals"]["0:0"]["matrix_count"], [2] * 9)
            self.assertEqual(merged["sample_counts"], {"natural": 2, "gaze": 1, "null": 1})

    def test_rejects_duplicate_e1_samples_across_ranks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = {
                "completed_sample_ids": ["duplicate"],
                "natural_totals": {},
                "gaze_totals": {},
                "sample_counts": {},
            }
            paths = [root / "rank0.json", root / "rank1.json"]
            for path in paths:
                path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate E1 sample"):
                merge_probe_checkpoints(paths, root / "merged.json")


if __name__ == "__main__":
    unittest.main()
