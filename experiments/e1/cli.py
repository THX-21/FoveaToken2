from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .analysis import analyze
from .config import E1Config
from .data import prepare_data
from .report import build_report
from .runner import run_all, run_probe


DEFAULT_CONFIG = Path(__file__).with_name("configs") / "default.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="E1 visual routing Head discovery")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="download natural images and build controlled grids")
    _config_argument(prepare)
    prepare.add_argument("--force", action="store_true", help="rebuild manifests and images")

    for command in ("probe", "analyze", "report", "run"):
        child = subparsers.add_parser(command)
        _config_argument(child)
        child.add_argument("--model", choices=("qwen25", "qwen35"), required=True)
        if command == "probe":
            child.add_argument("--pass", dest="pass_name", choices=("scan", "visualize"), default="scan")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = E1Config.load(args.config)
    if args.command == "prepare":
        counts = prepare_data(config, force=args.force)
        print(f"prepared {counts['natural']} natural and {counts['controlled']} controlled images")
    elif args.command == "probe":
        path = run_probe(config, args.model, pass_name=args.pass_name)
        print(path)
    elif args.command == "analyze":
        rows = analyze(config, args.model)
        print(f"analyzed {len(rows)} Heads in {config.output_dir / args.model}")
    elif args.command == "report":
        print(build_report(config, args.model))
    else:
        print(run_all(config, args.model))


def _config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
