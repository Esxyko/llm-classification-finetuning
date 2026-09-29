"""Command-line interface for selectable frozen-Qwen models."""

from __future__ import annotations

import argparse
import re
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

from ..cli_config import add_config_argument
from ..config import AppConfig
from ..errors import DataPreparationError
from . import MODEL_REGISTRY, ModelRegistration
from .common import (
    ModelBuildResult,
    ModelMode,
    ModelRunResult,
    ModelTestResult,
)
from .common.pipeline import ModelExecutionResult

ALL_MODELS = "ALL"


def _checkpoint_tag(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise argparse.ArgumentTypeError(
            "checkpoint tag must contain only letters, digits, underscores, or hyphens"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    """Create the shared model command-line parser."""
    parser = argparse.ArgumentParser(
        prog="model",
        description="Cross-validate, build, or test a frozen-Qwen comparison model.",
    )
    add_config_argument(parser)
    parser.add_argument(
        "model",
        choices=(*MODEL_REGISTRY, ALL_MODELS),
        help="Model profile to execute, or ALL to run every profile sequentially.",
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
        help="Load each selected model's saved head and generate a test submission.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Rebuild the selected mode's embedding cache for each selected model.",
    )
    parser.add_argument(
        "--checkpoint-tag",
        type=_checkpoint_tag,
        metavar="TAG",
        help="Save or load head_TAG.pt instead of head.pt (build and test only).",
    )
    return parser


def run(arguments: Sequence[str] | None = None) -> int:
    """Run the selected model mode and return a process exit code."""
    parsed_arguments = build_parser().parse_args(arguments)
    if parsed_arguments.checkpoint_tag and not (parsed_arguments.build or parsed_arguments.test):
        raise SystemExit("--checkpoint-tag requires --build or --test")
    try:
        config = AppConfig.load(parsed_arguments.config)
        project_root = Path.cwd()
        mode = _selected_mode(parsed_arguments)
        if parsed_arguments.model == ALL_MODELS:
            return _run_all_models(
                config=config,
                project_root=project_root,
                mode=mode,
                refresh_cache=parsed_arguments.refresh_cache,
                checkpoint_tag=parsed_arguments.checkpoint_tag,
            )

        registration = MODEL_REGISTRY[parsed_arguments.model]
        result = _run_registered_model(
            registration=registration,
            config=config,
            project_root=project_root,
            mode=mode,
            refresh_cache=parsed_arguments.refresh_cache,
            checkpoint_tag=parsed_arguments.checkpoint_tag,
        )
        _print_result(registration.profile.selector, result)
        return 0
    except DataPreparationError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


def _run_registered_model(
    *,
    registration: ModelRegistration,
    config: AppConfig,
    project_root: Path,
    mode: ModelMode,
    refresh_cache: bool,
    checkpoint_tag: str | None,
) -> ModelExecutionResult:
    return registration.pipeline_type(
        config=config,
        project_root=project_root,
        profile=registration.profile,
        checkpoint_tag=checkpoint_tag,
    ).run(mode=mode, refresh_cache=refresh_cache)


def _run_all_models(
    *,
    config: AppConfig,
    project_root: Path,
    mode: ModelMode,
    refresh_cache: bool,
    checkpoint_tag: str | None,
) -> int:
    """Run every registered model, preserving registry order after failures."""
    failed_models: list[str] = []
    for selector, registration in MODEL_REGISTRY.items():
        print(f"Running model: {selector}", flush=True)
        try:
            result = _run_registered_model(
                registration=registration,
                config=config,
                project_root=project_root,
                mode=mode,
                refresh_cache=refresh_cache,
                checkpoint_tag=checkpoint_tag,
            )
            _print_result(selector, result)
        except DataPreparationError as error:
            print(f"Error running {selector}: {error}", file=sys.stderr)
            failed_models.append(selector)
        except Exception as error:
            print(f"Error running {selector}:", file=sys.stderr)
            traceback.print_exception(error)
            failed_models.append(selector)

    completed = len(MODEL_REGISTRY) - len(failed_models)
    print(f"Completed {completed}/{len(MODEL_REGISTRY)} models.")
    if failed_models:
        print(f"Failed models: {', '.join(failed_models)}", file=sys.stderr)
    return int(bool(failed_models))


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
