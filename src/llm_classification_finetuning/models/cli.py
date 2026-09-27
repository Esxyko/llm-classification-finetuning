"""Command-line interface for selectable frozen-Qwen models."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

from ..config import AppConfig
from ..errors import DataPreparationError
from . import MODEL_REGISTRY
from .common import (
    ModelBuildResult,
    ModelMode,
    ModelRunResult,
    ModelTestResult,
)
from .common.pipeline import ModelExecutionResult

CONFIG_PATH = Path("config.yaml")


def build_parser() -> argparse.ArgumentParser:
    """Create the shared model command-line parser."""
    parser = argparse.ArgumentParser(
        prog="model",
        description="Cross-validate, build, or test a frozen-Qwen comparison model.",
    )
    parser.add_argument(
        "model",
        choices=tuple(MODEL_REGISTRY),
        help="Model profile to execute.",
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
        help="Load the selected model's saved head and generate a test submission.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore the selected mode's compatible embedding cache and rebuild it.",
    )
    return parser


def run(arguments: Sequence[str] | None = None) -> int:
    """Run the selected model mode and return a process exit code."""
    parsed_arguments = build_parser().parse_args(arguments)
    registration = MODEL_REGISTRY[parsed_arguments.model]
    profile = registration.profile
    try:
        config_path = CONFIG_PATH.resolve()
        config = AppConfig.load(config_path)
        result = registration.pipeline_type(
            config=config,
            project_root=config_path.parent,
            profile=profile,
        ).run(
            mode=_selected_mode(parsed_arguments),
            refresh_cache=parsed_arguments.refresh_cache,
        )
        _print_result(profile.selector, result)
        return 0
    except DataPreparationError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


def _selected_mode(arguments: argparse.Namespace) -> ModelMode:
    if arguments.build:
        return ModelMode.BUILD
    if arguments.test:
        return ModelMode.TEST
    return ModelMode.CROSS_VALIDATION


def _print_result(selector: str, result: ModelExecutionResult) -> None:
    cache_status = "Reused" if result.cache_reused else "Created"
    print(f"{cache_status} embedding cache: {result.cache_path}")
    if isinstance(result, ModelBuildResult):
        print(
            f"Trained the {selector} head on {result.training_rows:,} rows for "
            f"{result.epochs} epochs."
        )
        print(f"Final training loss: {result.final_training_loss:.6f}")
        print(f"Wrote model checkpoint: {result.checkpoint_path}")
    elif isinstance(result, ModelTestResult):
        print(f"Generated {result.predictions:,} competition test predictions.")
        print(f"Wrote model submission: {result.submission_path}")
    elif isinstance(result, ModelRunResult):
        print(
            f"Generated {result.predictions:,} out-of-fold predictions across "
            f"{result.folds} folds."
        )
        print(f"Average out-of-fold log loss: {result.average_loss:.6f}")
        print(f"Wrote model results: {result.output_dir}")
    else:
        raise TypeError(f"Unsupported model result: {type(result).__name__}")


def main() -> NoReturn:
    """Console-script entry point."""
    raise SystemExit(run())
