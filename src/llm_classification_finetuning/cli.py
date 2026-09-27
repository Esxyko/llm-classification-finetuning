"""Command-line interface for the project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NoReturn, Sequence

from dotenv import load_dotenv

from .cli_output import print_download_result, print_fold_result
from .config import AppConfig
from .errors import DataPreparationError
from .pipeline import PrepareDataPipeline

CONFIG_PATH = Path("config.yaml")


def build_parser() -> argparse.ArgumentParser:
    """Create the full initialization command-line parser."""
    parser = argparse.ArgumentParser(
        prog="init",
        description=(
            "Download competition data and generate folded, A/B-augmented "
            "training data."
        ),
    )
    return parser


def _prepare_data() -> None:
    resolved_config_path = CONFIG_PATH.resolve()
    load_dotenv(
        dotenv_path=resolved_config_path.parent / ".env",
        override=False,
    )
    config = AppConfig.load(resolved_config_path)
    result = PrepareDataPipeline(config).run(on_download=print_download_result)
    print_fold_result(result.folds)


def run(arguments: Sequence[str] | None = None) -> int:
    """Run the complete initialization workflow and return a process exit code."""
    build_parser().parse_args(arguments)

    try:
        _prepare_data()
        return 0
    except DataPreparationError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


def main() -> NoReturn:
    """Console-script entry point."""
    raise SystemExit(run())
