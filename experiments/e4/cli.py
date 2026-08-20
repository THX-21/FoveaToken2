from __future__ import annotations

import argparse

from .analysis import analyze
from .config import E4Config
from .data import prepare_data, prepare_external_assets
from .evaluator import load_tasks
from .report import build_report
from .runner import run

DEFAULT_CONFIG = "experiments/e4/configs/default.yaml"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="E4 formal dynamic multiscale evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run", "analyze", "report"):
        child = commands.add_parser(command)
        child.add_argument("--config", default=DEFAULT_CONFIG)
        if command != "prepare":
            child.add_argument("--model", choices=("qwen25", "qwen35"), required=True)
            child.add_argument(
                "--compression-ratio",
                type=float,
                help="override the configured high-res/active visual token ratio",
            )
        if command == "run":
            child.add_argument(
                "--suite",
                choices=("formal", "reasoning", "compression"),
                default="formal",
            )
            child.add_argument("--condition")
            child.add_argument("--task")
        if command in {"analyze", "report"}:
            child.add_argument(
                "--suite",
                choices=("formal", "reasoning", "compression"),
                action="append",
            )
    args = parser.parse_args(argv)
    config = E4Config.load(args.config)
    if args.command != "prepare" and args.compression_ratio is not None:
        config.compression_ratio = args.compression_ratio
        config.compression_ratios = (args.compression_ratio,)
        config.validate()
    if args.command == "prepare":
        tasks = load_tasks("qwen25", config.formal_tasks)
        manifest = prepare_data(config, tasks)
        assets = prepare_external_assets(config)
        print(f"{manifest}\n{assets}")
    elif args.command == "run":
        print(
            run(
                config,
                args.model,
                args.suite,
                condition_name=args.condition,
                task_name=args.task,
            )
        )
    elif args.command == "analyze":
        rows = analyze(config, args.model, suites=args.suite)
        print(f"wrote {len(rows)} E4 summary rows")
    else:
        print(build_report(config, args.model, suites=args.suite))
