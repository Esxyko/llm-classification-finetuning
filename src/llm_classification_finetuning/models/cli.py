"""Command-line interface for selectable frozen and LoRA Qwen models."""

from __future__ import annotations

import argparse
import re
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import NoReturn

from ..cli_config import add_config_argument
from ..config import AppConfig
from ..errors import ConfigurationError, DataPreparationError
from ..timing import format_duration
from . import MODEL_REGISTRY, ModelRegistration
from .common import (
	ModelBuildResult,
	ModelExecutionResult,
	ModelMode,
	ModelResultPublisher,
	ModelRunResult,
	ModelTestResult,
)

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
		description="Cross-validate, build, or test a Qwen comparison model.",
	)
	add_config_argument(parser)
	parser.add_argument(
		"model",
		choices=(*MODEL_REGISTRY, ALL_MODELS),
		help="Model profile to execute, or ALL to run every configured profile sequentially.",
	)
	modes = parser.add_mutually_exclusive_group()
	modes.add_argument(
		"--build",
		action="store_true",
		help="Train on all folds and save the classifier head and any LoRA adapters.",
	)
	modes.add_argument(
		"--test",
		action="store_true",
		help="Load each selected model's checkpoint and generate a test submission.",
	)
	parser.add_argument(
		"--refresh-cache",
		action="store_true",
		help="Rebuild frozen-model embeddings; has no effect for LoRA models.",
	)
	parser.add_argument(
		"--checkpoint-tag",
		type=_checkpoint_tag,
		metavar="TAG",
		help="Select head_TAG.pt or adapter_TAG.pt (build and test only).",
	)
	return parser


def run(arguments: Sequence[str] | None = None) -> int:
	"""Run the selected model mode and return a process exit code."""
	parsed_arguments = build_parser().parse_args(arguments)
	if parsed_arguments.checkpoint_tag and not (
		parsed_arguments.build or parsed_arguments.test
	):
		raise SystemExit("--checkpoint-tag requires --build or --test")
	try:
		config = AppConfig.load(parsed_arguments.config)
		project_root = Path.cwd()
		mode = _selected_mode(parsed_arguments)
		if parsed_arguments.model == ALL_MODELS:
			return _run_all_models(
				config=config,
				config_path=parsed_arguments.config,
				project_root=project_root,
				mode=mode,
				refresh_cache=parsed_arguments.refresh_cache,
				checkpoint_tag=parsed_arguments.checkpoint_tag,
			)

		registration = MODEL_REGISTRY[parsed_arguments.model]
		result = _run_registered_model(
			registration=registration,
			config=config,
			config_path=parsed_arguments.config,
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
	config_path: Path,
	project_root: Path,
	mode: ModelMode,
	refresh_cache: bool,
	checkpoint_tag: str | None,
) -> ModelExecutionResult:
	model_started = perf_counter()
	result = registration.pipeline_type(
		config=config,
		project_root=project_root,
		profile=registration.profile,
		checkpoint_tag=checkpoint_tag,
	).run(mode=mode, refresh_cache=refresh_cache)
	elapsed_seconds = perf_counter() - model_started
	if isinstance(result, ModelRunResult):
		ModelResultPublisher.record_model_time(
			result.output_dir, config_path, elapsed_seconds
		)
	print(
		f"Model total time ({registration.profile.selector}): "
		f"{format_duration(elapsed_seconds)}"
	)
	return result


def _run_all_models(
	*,
	config: AppConfig,
	config_path: Path,
	project_root: Path,
	mode: ModelMode,
	refresh_cache: bool,
	checkpoint_tag: str | None,
) -> int:
	"""Run configured models, preserving registry order after failures."""
	registrations = {
		selector: registration
		for selector, registration in MODEL_REGISTRY.items()
		if registration.profile.is_configured(config)
	}
	if not registrations:
		raise ConfigurationError(
			"The selected configuration contains no model sections."
		)
	failed_models: list[str] = []
	for selector, registration in registrations.items():
		print(f"Running model: {selector}", flush=True)
		try:
			result = _run_registered_model(
				registration=registration,
				config=config,
				config_path=config_path,
				project_root=project_root,
				mode=mode,
				refresh_cache=refresh_cache,
				checkpoint_tag=checkpoint_tag,
			)
			_print_result(selector, result)
		except DataPreparationError as error:
			print(f"Error running {selector}: {error}", file=sys.stderr)
			failed_models.append(selector)
		except Exception as error:  # noqa: BLE001 - continue after unexpected model failures
			print(f"Error running {selector}:", file=sys.stderr)
			traceback.print_exception(error)
			failed_models.append(selector)

	completed = len(registrations) - len(failed_models)
	print(f"Completed {completed}/{len(registrations)} models.")
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
	if result.cache_path is not None:
		cache_status = "Reused" if result.cache_reused else "Created"
		print(f"{cache_status} embedding cache: {result.cache_path}")
	if isinstance(result, ModelBuildResult):
		print(
			f"Trained {selector} on {result.training_rows:,} rows for "
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
