from __future__ import annotations

import argparse

from .analysis import analyze
from .config import E3Config
from .data import prepare_data
from .report import build_report
from .runner import reevaluate, run
from .evaluator import SCORING_VERSION

DEFAULT_CONFIG = "experiments/e3/configs/default.yaml"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="E3 Text-Anchor position experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run", "reevaluate", "analyze", "report"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", default=DEFAULT_CONFIG)
        if command != "prepare":
            child.add_argument("--model", choices=("qwen25", "qwen35"), required=True)
        if command in ("run", "reevaluate"):
            child.add_argument(
                "--version",
                default=SCORING_VERSION,
                help=f"score version recorded in results (default: {SCORING_VERSION})",
            )
            child.add_argument("--condition")
        if command == "reevaluate":
            child.add_argument(
                "--mode",
                choices=("resume", "restart"),
                default="resume",
                help="score only samples missing this version (default), or score all samples again",
            )
            child.add_argument(
                "--workers",
                type=int,
                default=8,
                help="maximum concurrent Judge requests (default: 8)",
            )
    args = parser.parse_args(argv)
    config = E3Config.load(args.config)
    if args.command == "prepare":
        print(prepare_data(config))
    elif args.command == "run":
        print(
            run(
                config,
                args.model,
                condition_name=args.condition,
                scoring_version=args.version,
            )
        )
    elif args.command == "reevaluate":
        print(
            reevaluate(
                config,
                args.model,
                condition_name=args.condition,
                scoring_version=args.version,
                restart=args.mode == "restart",
                workers=args.workers,
            )
        )
    elif args.command == "analyze":
        rows = analyze(config, args.model)
        print(f"wrote {len(rows)} E3 summary rows")
    else:
        print(build_report(config, args.model))
