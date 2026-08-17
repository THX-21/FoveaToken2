import unittest
from unittest.mock import patch

from experiments.e3.cli import main
from experiments.e3.evaluator import SCORING_VERSION


class E3CliTest(unittest.TestCase):
    def test_run_uses_the_default_score_version(self):
        with patch("experiments.e3.cli.run") as run:
            main(["run", "--model", "qwen25"])

        self.assertEqual(run.call_args.kwargs["scoring_version"], SCORING_VERSION)

    def test_reevaluate_passes_version_and_mode(self):
        with patch("experiments.e3.cli.reevaluate") as reevaluate:
            main([
                "reevaluate",
                "--model",
                "qwen25",
                "--version",
                "judge_1k_v2",
                "--mode",
                "restart",
            ])

        self.assertEqual(reevaluate.call_args.kwargs["scoring_version"], "judge_1k_v2")
        self.assertTrue(reevaluate.call_args.kwargs["restart"])
        self.assertEqual(reevaluate.call_args.kwargs["workers"], 8)


if __name__ == "__main__":
    unittest.main()
