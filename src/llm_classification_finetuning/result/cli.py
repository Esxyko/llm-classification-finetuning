"""Command-line interface for synthesizing cross-validation results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NoReturn, Sequence

from ..cli_config import add_config_argument
from ..config import AppConfig
from ..errors import DataPreparationError
from .synthesizer import ResultSynthesisResult, ResultSynthesizer


def build_parser() -> argparse.ArgumentParser:
    """Create the comprehensive-result command-line parser."""
    parser = argparse.ArgumentParser(
        prog="result",
        description=(
            "Synthesize k-fold prediction CSVs into a comprehensive workbook "
            "and confusion matrix."
        ),
    )
    add_config_argument(parser)
    parser.add_argument(
        "subfolder",
        nargs="?",
        help=(
            "Direct child of results containing fold CSVs. Defaults to the "
            "most recently modified result subfolder."
        ),
    )
    return parser


def run(arguments: Sequence[str] | None = None) -> int:
    """Synthesize a result run and return a process exit code."""
    parsed_arguments = build_parser().parse_args(arguments)

    try:
        config = AppConfig.load(parsed_arguments.config)
        result = ResultSynthesizer(
            processed_path=config.data.processed_path,
            n_splits=config.cross_validation.n_splits,
            results_root=Path.cwd() / "results",
        ).synthesize(parsed_arguments.subfolder)
        _print_result(result)
        return 0
    except DataPreparationError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


def _print_result(result: ResultSynthesisResult) -> None:
    print(f"Selected result run: {result.source_dir}")
    print(
        f"Synthesized {result.predictions:,} predictions from {result.folds} "
        f"folds into {result.records:,} records."
    )
    print(f"Incorrect predictions: {result.incorrects:,}")
    print(f"Average log loss: {result.average_loss:.6f}")
    print(f"Wrote records workbook: {result.workbook_path}")
    print(f"Wrote confusion matrix: {result.confusion_matrix_path}")


def main() -> NoReturn:
    """Console-script entry point."""
    raise SystemExit(run())
