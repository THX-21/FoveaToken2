import unittest
from types import SimpleNamespace

from experiments.e2.config import E2Config
from experiments.e2.evaluator import _chat_template, _max_new_tokens


class E2ThinkingTest(unittest.TestCase):
    def test_passes_thinking_to_qwen35_template(self):
        processor = SimpleNamespace()
        captured = {}

        def apply_chat_template(messages, **kwargs):
            captured.update(kwargs)
            return "prompt"

        processor.apply_chat_template = apply_chat_template
        lm = SimpleNamespace(processor=processor)

        self.assertEqual(_chat_template(lm, [], "qwen35", thinking=True), "prompt")
        self.assertTrue(captured["enable_thinking"])

    def test_uses_2048_tokens_with_thinking(self):
        config = E2Config(max_new_tokens=16)

        self.assertEqual(_max_new_tokens(config, {"max_new_tokens": 16}, thinking=True), 2048)
        self.assertEqual(_max_new_tokens(config, {"max_new_tokens": 32}, thinking=False), 16)


if __name__ == "__main__":
    unittest.main()
