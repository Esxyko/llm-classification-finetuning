"""Command-line interface for the Qwen3 pairwise baseline."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

from ..config import AppConfig
from ..errors import DataPreparationError
from .pipeline import (
    BaselineBuildResult,
    BaselineExecutionResult,
    BaselineMode,
    BaselinePipeline,
    BaselineRunResult,
    BaselineTestResult,
)

CONFIG_PATH = Path("config.yaml")


def build_parser() -> argparse.ArgumentParser:
    """Create the baseline command-line parser."""
    parser = argparse.ArgumentParser(
        prog="baseline",
        description=(
            "Cross-validate, build, or test the frozen-Qwen3 pairwise baseline."
        ),
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--build",
        action="store_true",
        help="Train one head on all folds and save its parameters.",
    )
    modes.add_argument(
        "--test",
        action="store_true",
        help="Load the saved head and generate a test submission without training.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore the selected mode's compatible embedding cache and rebuild it.",
    )
    return parser


def run(arguments: Sequence[str] | None = None) -> int:
    """Run the selected baseline mode and return a process exit code."""
    parsed_arguments = build_parser().parse_args(arguments)
    try:
        config_path = CONFIG_PATH.resolve()
        config = AppConfig.load(config_path)
        result = BaselinePipeline(
            config=config,
            project_root=config_path.parent,
        ).run(
            mode=_selected_mode(parsed_arguments),
            refresh_cache=parsed_arguments.refresh_cache,
        )
        _print_result(result)
        return 0
    except DataPreparationError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


def _selected_mode(arguments: argparse.Namespace) -> BaselineMode:
    if arguments.build:
        return BaselineMode.BUILD
    if arguments.test:
        return BaselineMode.TEST
    return BaselineMode.CROSS_VALIDATION


def _print_result(result: BaselineExecutionResult) -> None:
    cache_status = "Reused" if result.cache_reused else "Created"
    print(f"{cache_status} embedding cache: {result.cache_path}")
    if isinstance(result, BaselineBuildResult):
        print(
            f"Trained the baseline head on {result.training_rows:,} rows for "
            f"{result.epochs} epochs."
        )
        print(f"Final training loss: {result.final_training_loss:.6f}")
        print(f"Wrote baseline checkpoint: {result.checkpoint_path}")
    elif isinstance(result, BaselineTestResult):
        print(f"Generated {result.predictions:,} competition test predictions.")
        print(f"Wrote baseline submission: {result.submission_path}")
    elif isinstance(result, BaselineRunResult):
        print(
            f"Generated {result.predictions:,} out-of-fold predictions across "
            f"{result.folds} folds."
        )
        print(f"Average out-of-fold log loss: {result.average_loss:.6f}")
        print(f"Wrote baseline results: {result.output_dir}")
    else:
        raise TypeError(f"Unsupported baseline result: {type(result).__name__}")


def main() -> NoReturn:
    """Console-script entry point."""
    raise SystemExit(run())
