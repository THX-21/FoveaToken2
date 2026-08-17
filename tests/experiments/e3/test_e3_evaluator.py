import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from experiments.e2.image import ImagePlan
from experiments.e3.evaluator import (
    _ground_truth,
    _messages,
    _prepare_inputs,
    _read_completed_samples,
    _score_response,
    build_analysis_prompt,
    parse_response,
)


class E3EvaluatorTest(unittest.TestCase):
    def test_prompt_removes_the_conflicting_task_instruction(self):
        prompt = build_analysis_prompt("What is shown?\nAnswer the question with a single word.")

        self.assertEqual(prompt, "What is shown?")

        prompt = build_analysis_prompt("What is shown?\nAnswer the question using a single word or phrase.")

        self.assertEqual(prompt, "What is shown?")

    def test_parser_scores_only_answer_and_counts_analyze_words(self):
        parsed = parse_response(
            "<analysis>The red sign clearly displays a large white stop label.</analysis>\n"
            "<answer>stop</answer>"
        )
        self.assertTrue(parsed.format_compliant)
        self.assertEqual(parsed.answer, "stop")
        self.assertEqual(parsed.analyze_word_count, 10)
        self.assertTrue(parsed.analyze_within_limit)

    def test_parser_rejects_an_untagged_answer(self):
        parsed = parse_response("The sign has a blue border. Answer: blue")

        self.assertFalse(parsed.format_compliant)
        self.assertEqual(parsed.answer, "")

    def test_parser_accepts_the_lmms_think_tags(self):
        parsed = parse_response("<think>inspect the sign</think><answer>blue</answer>")

        self.assertTrue(parsed.format_compliant)
        self.assertEqual(parsed.answer, "blue")

    def test_parser_rejects_a_fallback_answer(self):
        parsed = parse_response("The image supports the answer.\nblue")
        self.assertFalse(parsed.format_compliant)
        self.assertEqual(parsed.answer, "")

    def test_messages_use_the_lmms_reasoning_system_prompt(self):
        messages = _messages(SimpleNamespace(system_prompt="ignored"), Image.new("RGB", (1, 1)), "question")

        self.assertEqual(
            messages[0]["content"][0]["text"],
            "You are a helpful assistant. When the user asks a question, your response must include two "
            "parts: first, the reasoning process enclosed in `<analysis>...</analysis>` tags, followed by "
            "a clear, concise final answer enclosed in `<answer>...</answer>` tags that directly addresses "
            "the question and contains only the short final answer without explanation.",
        )
        self.assertEqual(messages[1]["content"][1]["text"], "question")

    def test_reasoning_score_uses_lmms_parser_and_task_ground_truth(self):
        response = "<analysis>The answer is visible.</analysis><answer>yes</answer>"
        score = _score_response(
            "vqav2_val_lite",
            {"multiple_choice_answer": "yes"},
            "Is this true?",
            response,
        )

        self.assertEqual(score, {"exact_match": 1.0, "format_score": 1.0})
        self.assertEqual(_ground_truth("textvqa_val_lite", {"answers": ["7", "7", "1"]}), "7")

    def test_reasoning_score_judges_invalid_format(self):
        with (
            patch("experiments.e3.evaluator.USE_LLM_JUDGE", "True"),
            patch("experiments.e3.evaluator.llm_as_judge_sync", return_value=1) as judge,
        ):
            score = _score_response(
                "gqa_lite", {"answer": "man"}, "Who is shown?", "The answer is man."
            )

        self.assertEqual(score, {"exact_match": 1.0, "format_score": 0.0})
        judge.assert_called_once_with("The answer is man.", "man", {"question": "Who is shown?"})

    def test_reads_completed_samples_for_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.jsonl"
            first = {"sample_id": "vqav2_val_lite:4", "metrics": {"exact_match": 1}}
            path.write_text(json.dumps(first) + "\n{", encoding="utf-8")

            completed = _read_completed_samples(path, {"vqav2_val_lite:4", "vqav2_val_lite:7"})

            self.assertEqual(completed, {"vqav2_val_lite:4": first})

    def test_prepared_inputs_validate_image_token_count(self):
        class Batch(dict):
            def to(self, _device):
                return self

        class Processor:
            def apply_chat_template(self, _messages, **_kwargs):
                return "prompt"

            def __call__(self, **_kwargs):
                return Batch(
                    input_ids=torch.tensor([[1, 99, 99, 99, 2]]),
                    image_grid_thw=torch.tensor([[1, 4, 4]]),
                )

        model = SimpleNamespace(
            config=SimpleNamespace(
                image_token_id=99,
                vision_config=SimpleNamespace(spatial_merge_size=2),
            )
        )
        lm = SimpleNamespace(processor=Processor(), model=model, device="cpu")
        plan = ImagePlan(width=4, height=4, grid_width=2, grid_height=2)

        with self.assertRaisesRegex(ValueError, "image token count"):
            _prepare_inputs(lm, "qwen25", Image.new("RGB", (4, 4)), "question", plan)


if __name__ == "__main__":
    unittest.main()
