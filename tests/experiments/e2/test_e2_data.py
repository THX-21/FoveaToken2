import unittest

from experiments.e2.config import E2Config, ModelSpec
from experiments.e2.data import validate_manifest


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


if __name__ == "__main__":
    unittest.main()
