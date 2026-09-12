"""Command-line interface for generating folded, A/B-augmented data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NoReturn, Sequence

from ..config import AppConfig
from ..errors import DataPreparationError
from .folds import FoldPreparationResult, FoldPreprocessor

CONFIG_PATH = Path("config.yaml")


def build_parser() -> argparse.ArgumentParser:
    """Create the training-data preprocessing command-line parser."""
    parser = argparse.ArgumentParser(
        prog="preprocess",
        description=(
            "Generate grouped cross-validation folds and A/B augmentation from "
            "raw training data."
        ),
    )
    return parser


def run(arguments: Sequence[str] | None = None) -> int:
    """Generate the processed artifact and return a process exit code."""
    build_parser().parse_args(arguments)

    try:
        config = AppConfig.load(CONFIG_PATH.resolve())
        result = FoldPreprocessor(config.cross_validation).prepare(
            train_path=config.data.raw_dir / "train.csv",
            output_path=config.data.processed_path,
        )
        _print_result(result)
        return 0
    except DataPreparationError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


def _print_result(result: FoldPreparationResult) -> None:
    print(f"Prepared {result.rows:,} rows across {result.groups:,} prompt groups.")
    print("Fold distribution:")
    print("fold  rows     groups   model_a   model_b   tie")
    for summary in result.folds:
        class_percentages = tuple(
            count / summary.rows * 100 for count in summary.class_counts
        )
        print(
            f"{summary.fold:>4}  {summary.rows:>8,}  {summary.groups:>7,}  "
            f"{class_percentages[0]:>7.2f}%  "
            f"{class_percentages[1]:>7.2f}%  "
            f"{class_percentages[2]:>6.2f}%"
        )
    print(f"Wrote processed data: {result.output_path}")


def main() -> NoReturn:
    """Console-script entry point."""
    raise SystemExit(run())
