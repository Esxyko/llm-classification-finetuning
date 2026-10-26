"""Command-line interface for generating holdout-split, A/B-augmented data."""

from __future__ import annotations

import argparse
import sys
from typing import NoReturn, Sequence

from ..cli_config import add_config_argument
from ..cli_output import print_split_result
from ..config import AppConfig
from ..errors import DataPreparationError
from .splits import SplitPreprocessor


def build_parser() -> argparse.ArgumentParser:
	"""Create the training-data preprocessing command-line parser."""
	parser = argparse.ArgumentParser(
		prog="preprocess",
		description=(
			"Generate a grouped train/validation split and A/B augmentation from "
			"raw training data."
		),
	)
	add_config_argument(parser)
	return parser


def run(arguments: Sequence[str] | None = None) -> int:
	"""Generate the processed artifact and return a process exit code."""
	parsed_arguments = build_parser().parse_args(arguments)

	try:
		config = AppConfig.load(parsed_arguments.config)
		result = SplitPreprocessor(
			config.validation, cache_dir=config.cache_dir
		).prepare(
			train_path=config.data.raw_dir / "train.csv",
			output_path=config.data.processed_path,
		)
		print_split_result(result)
		return 0
	except DataPreparationError as error:
		print(f"Error: {error}", file=sys.stderr)
		return 1


def main() -> NoReturn:
	"""Console-script entry point."""
	raise SystemExit(run())
