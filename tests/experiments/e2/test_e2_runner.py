import sys
import types
import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
