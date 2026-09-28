"""Command-line interface for downloading raw competition data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NoReturn, Sequence

from dotenv import load_dotenv

from ..cli_config import add_config_argument
from ..cli_output import print_download_result
from ..config import AppConfig
from ..errors import DataPreparationError
from .downloader import CompetitionDataDownloader


def build_parser() -> argparse.ArgumentParser:
    """Create the raw-data download command-line parser."""
    parser = argparse.ArgumentParser(
        prog="data",
        description="Download missing Kaggle competition data files.",
    )
    add_config_argument(parser)
    return parser


def run(arguments: Sequence[str] | None = None) -> int:
    """Download missing raw files and return a process exit code."""
    parsed_arguments = build_parser().parse_args(arguments)

    try:
        load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
        config = AppConfig.load(parsed_arguments.config)
        result = CompetitionDataDownloader(config.data).ensure_available()
        print_download_result(result)
        return 0
    except DataPreparationError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


def main() -> NoReturn:
    """Console-script entry point."""
    raise SystemExit(run())
