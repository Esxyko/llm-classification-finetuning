"""Shared command-line option for selecting application configuration."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    """Allow a command to load a complete YAML config from a selected path."""
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        metavar="PATH",
        help="YAML configuration file (default: config.yaml in the working directory).",
    )
