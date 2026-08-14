from __future__ import annotations

import argparse

from .analysis import analyze
from .config import E3Config
from .data import prepare_data
from .report import build_report
from .runner import run

DEFAULT_CONFIG = "experiments/e3/configs/default.yaml"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="E3 Text-Anchor position experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run", "analyze", "report"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", default=DEFAULT_CONFIG)
        if command != "prepare":
            child.add_argument("--model", choices=("qwen25", "qwen35"), required=True)
        if command == "run":
            child.add_argument("--condition")
    args = parser.parse_args(argv)
    config = E3Config.load(args.config)
    if args.command == "prepare":
        print(prepare_data(config))
    elif args.command == "run":
        print(run(config, args.model, condition_name=args.condition))
    elif args.command == "analyze":
        rows = analyze(config, args.model)
        print(f"wrote {len(rows)} E3 summary rows")
    else:
        print(build_report(config, args.model))
