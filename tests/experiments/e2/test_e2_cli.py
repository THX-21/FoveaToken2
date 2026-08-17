import unittest
from unittest.mock import patch

from experiments.e2.cli import main


class E2CliTest(unittest.TestCase):
    def test_prepare_does_not_require_model_arguments(self):
        with patch("experiments.e2.cli.prepare_data", return_value="manifest.json") as prepare:
            main(["prepare"])

        prepare.assert_called_once()


if __name__ == "__main__":
    unittest.main()
