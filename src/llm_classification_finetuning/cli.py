"""Command-line interface for the project."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

from .cli_config import add_config_argument
from .cli_output import print_download_result, print_split_result
from .config import AppConfig
from .errors import DataPreparationError
from .pipeline import PrepareDataPipeline


def build_parser() -> argparse.ArgumentParser:
	"""Create the full initialization command-line parser."""
	parser = argparse.ArgumentParser(
		prog="init",
		description=(
			"Download competition data and generate holdout-split, A/B-augmented "
			"training data."
		),
	)
	add_config_argument(parser)
	return parser


def _prepare_data(config_path: Path) -> None:
	config = AppConfig.load(config_path)
	result = PrepareDataPipeline(config).run(on_download=print_download_result)
	print_split_result(result.splits)


def run(arguments: Sequence[str] | None = None) -> int:
	"""Run the complete initialization workflow and return a process exit code."""
	parsed_arguments = build_parser().parse_args(arguments)

	try:
		_prepare_data(parsed_arguments.config)
		return 0
	except DataPreparationError as error:
		print(f"Error: {error}", file=sys.stderr)
		return 1


def main() -> NoReturn:
	"""Console-script entry point."""
	raise SystemExit(run())
