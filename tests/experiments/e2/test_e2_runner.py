import sys
import types
import unittest
from unittest.mock import patch

from experiments.distributed import DistributedContext
from experiments.e2.config import ModelSpec
from experiments.e2.evaluator import _lmms_model_name
from experiments.e2.runner import DEVICE, load_lm, run_name


class E2RunnerTest(unittest.TestCase):
    def test_uses_the_qwen25_lmms_backend_name(self):
        self.assertEqual(_lmms_model_name("qwen25"), "qwen2_5_vl")

    def test_loads_model_on_one_visible_gpu(self):
        captured = {}

        class FakeModel:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        module = types.ModuleType("lmms_eval.models.simple.qwen3_5")
        module.Qwen3_5 = FakeModel

        with patch.dict(sys.modules, {module.__name__: module}):
            load_lm(ModelSpec("fake", 1, 2, 32), "qwen35")

        self.assertEqual(captured["device"], DEVICE)
        self.assertEqual(captured["device_map"], DEVICE)

    def test_enables_qwen35_thinking(self):
        captured = {}

        class FakeModel:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        module = types.ModuleType("lmms_eval.models.simple.qwen3_5")
        module.Qwen3_5 = FakeModel

        with patch.dict(sys.modules, {module.__name__: module}):
            load_lm(ModelSpec("fake", 1, 2, 32), "qwen35", thinking=True)

        self.assertTrue(captured["enable_thinking"])
        self.assertEqual(run_name("qwen35", thinking=True), "qwen35_thinking")

    def test_distributed_worker_loads_model_on_its_local_gpu(self):
        captured = {}

        class FakeModel:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        module = types.ModuleType("lmms_eval.models.simple.qwen3_5")
        module.Qwen3_5 = FakeModel
        context = DistributedContext(rank=1, local_rank=2, world_size=4)

        with (
            patch.dict(sys.modules, {module.__name__: module}),
            patch("experiments.e2.runner.distributed_context", return_value=context),
        ):
            load_lm(ModelSpec("fake", 1, 2, 32), "qwen35")

        self.assertEqual(captured["device"], "cuda:2")
        self.assertEqual(captured["device_map"], "cuda:2")


if __name__ == "__main__":
    unittest.main()
