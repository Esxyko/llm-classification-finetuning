"""Load and validate application configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Locations and competition identity used by the data pipeline."""

    competition: str
    raw_dir: Path
    processed_path: Path


@dataclass(frozen=True, slots=True)
class CrossValidationConfig:
    """Parameters controlling deterministic fold generation."""

    n_splits: int
    random_state: int


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Validated application configuration."""

    data: DataConfig
    cross_validation: CrossValidationConfig

    @classmethod
    def load(cls, config_path: Path) -> AppConfig:
        """Read a YAML configuration and resolve paths relative to it."""
        resolved_config_path = config_path.expanduser().resolve()
        if not resolved_config_path.is_file():
            raise ConfigurationError(
                f"Configuration file not found: {resolved_config_path}"
            )

        try:
            with resolved_config_path.open(encoding="utf-8") as config_file:
                raw_config = yaml.safe_load(config_file)
        except yaml.YAMLError as error:
            raise ConfigurationError(
                f"Invalid YAML in {resolved_config_path}: {error}"
            ) from error
        except OSError as error:
            raise ConfigurationError(
                f"Could not read configuration {resolved_config_path}: {error}"
            ) from error

        root = _require_mapping(raw_config, "configuration root")
        _reject_unknown_keys(root, {"data", "cross_validation"}, "configuration root")

        data_section = _require_mapping(root.get("data"), "data")
        _reject_unknown_keys(
            data_section,
            {"competition", "raw_dir", "processed_path"},
            "data",
        )
        cross_validation_section = _require_mapping(
            root.get("cross_validation"), "cross_validation"
        )
        _reject_unknown_keys(
            cross_validation_section,
            {"n_splits", "random_state"},
            "cross_validation",
        )

        competition = _require_non_empty_string(
            data_section.get("competition"), "data.competition"
        )
        base_dir = resolved_config_path.parent
        raw_dir = _resolve_config_path(
            base_dir,
            _require_non_empty_string(data_section.get("raw_dir"), "data.raw_dir"),
        )
        processed_path = _resolve_config_path(
            base_dir,
            _require_non_empty_string(
                data_section.get("processed_path"), "data.processed_path"
            ),
        )
        if processed_path.suffix.lower() != ".parquet":
            raise ConfigurationError(
                "data.processed_path must point to a .parquet file."
            )

        n_splits = _require_integer(
            cross_validation_section.get("n_splits"),
            "cross_validation.n_splits",
        )
        if n_splits < 2:
            raise ConfigurationError("cross_validation.n_splits must be at least 2.")

        random_state = _require_integer(
            cross_validation_section.get("random_state"),
            "cross_validation.random_state",
        )

        return cls(
            data=DataConfig(
                competition=competition,
                raw_dir=raw_dir,
                processed_path=processed_path,
            ),
            cross_validation=CrossValidationConfig(
                n_splits=n_splits,
                random_state=random_state,
            ),
        )


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{field_name} must be a YAML mapping.")
    return value


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _require_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{field_name} must be an integer.")
    return value


def _resolve_config_path(base_dir: Path, configured_path: str) -> Path:
    path = Path(configured_path).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _reject_unknown_keys(
    section: Mapping[str, Any], allowed_keys: set[str], section_name: str
) -> None:
    unknown_keys = sorted(set(section) - allowed_keys)
    if unknown_keys:
        keys = ", ".join(unknown_keys)
        raise ConfigurationError(f"Unknown key(s) in {section_name}: {keys}")
