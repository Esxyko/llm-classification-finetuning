"""Command-line interface for downloading raw competition data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NoReturn, Sequence

from dotenv import load_dotenv

from ..config import AppConfig
from ..errors import DataPreparationError
from .downloader import CompetitionDataDownloader, DownloadResult

CONFIG_PATH = Path("config.yaml")


def build_parser() -> argparse.ArgumentParser:
    """Create the raw-data download command-line parser."""
    parser = argparse.ArgumentParser(
        prog="data",
        description="Download missing Kaggle competition data files.",
    )
    return parser


def run(arguments: Sequence[str] | None = None) -> int:
    """Download missing raw files and return a process exit code."""
    build_parser().parse_args(arguments)

    try:
        config_path = CONFIG_PATH.resolve()
        load_dotenv(dotenv_path=config_path.parent / ".env", override=False)
        config = AppConfig.load(config_path)
        result = CompetitionDataDownloader(config.data).ensure_available()
        _print_result(result)
        return 0
    except DataPreparationError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


def _print_result(result: DownloadResult) -> None:
    for file_name in result.skipped:
        print(f"Skipped existing raw file: {file_name}")
    for file_name in result.downloaded:
        print(f"Downloaded raw file: {file_name}")


def main() -> NoReturn:
    """Console-script entry point."""
    raise SystemExit(run())
