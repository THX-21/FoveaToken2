import unittest

from experiments.e3.evaluator import build_analysis_prompt, parse_response


class E3EvaluatorTest(unittest.TestCase):
    def test_prompt_uses_natural_analyze_answer_format(self):
        prompt = build_analysis_prompt("What is shown?")
        self.assertIn("do not exceed 200 words", prompt)
        self.assertIn("Analyze: <your analysis>", prompt)
        self.assertIn("Answer: <your final answer>", prompt)

    def test_parser_scores_only_answer_and_counts_analyze_words(self):
        parsed = parse_response(
            "Analyze: The red sign clearly displays a large white stop label.\n"
            "Answer: stop"
        )
        self.assertTrue(parsed.format_compliant)
        self.assertEqual(parsed.answer, "stop")
        self.assertEqual(parsed.analyze_word_count, 10)
        self.assertTrue(parsed.analyze_within_limit)

    def test_parser_has_a_safe_fallback(self):
        parsed = parse_response("The image supports the answer.\nblue")
        self.assertFalse(parsed.format_compliant)
        self.assertEqual(parsed.answer, "blue")


if __name__ == "__main__":
    unittest.main()
