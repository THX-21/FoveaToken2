from __future__ import annotations

import argparse

from .analysis import analyze
from .config import E2Config
from .data import prepare_data
from .report import build_report
from .runner import run

DEFAULT_CONFIG = "experiments/e2/configs/default.yaml"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="E2 coarse visual representation experiment")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run", "analyze", "report"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", default=DEFAULT_CONFIG)
        if command != "prepare":
            child.add_argument("--model", choices=("qwen25", "qwen35"), required=True)
            child.add_argument("--thinking", action="store_true", help="enable Qwen3.5 thinking (qwen35 only)")
        if command == "run":
            child.add_argument("--condition")
    args = parser.parse_args(argv)
    if args.command != "prepare" and args.thinking and args.model != "qwen35":
        parser.error("--thinking is only supported for qwen35")
    config = E2Config.load(args.config)
    if args.command == "prepare":
        print(prepare_data(config))
    elif args.command == "run":
        print(run(config, args.model, condition_name=args.condition, thinking=args.thinking))
    elif args.command == "analyze":
        rows = analyze(config, args.model, thinking=args.thinking)
        print(f"wrote {len(rows)} E2 summary rows")
    else:
        print(build_report(config, args.model, thinking=args.thinking))
