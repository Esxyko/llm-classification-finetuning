"""Load and validate application configuration."""

from __future__ import annotations

import math
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
class GPUConfig:
    """Project-wide settings for one CUDA device."""

    device: str
    precision: str
    attention_implementation: str
    allow_tf32: bool


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """Hyperparameters for the frozen-backbone pairwise baseline."""

    model_name: str
    max_length: int
    extraction_batch_size: int
    training_batch_size: int
    hidden_size: int
    dropout: float
    epochs: int
    learning_rate: float
    weight_decay: float
    random_state: int
    dataloader_workers: int


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Validated application configuration."""

    data: DataConfig
    cross_validation: CrossValidationConfig
    gpu: GPUConfig
    baseline: BaselineConfig

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
        _reject_unknown_keys(
            root,
            {"data", "cross_validation", "gpu", "baseline"},
            "configuration root",
        )

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
        gpu_section = _require_mapping(root.get("gpu"), "gpu")
        _reject_unknown_keys(
            gpu_section,
            {"device", "precision", "attention_implementation", "allow_tf32"},
            "gpu",
        )
        baseline_section = _require_mapping(root.get("baseline"), "baseline")
        _reject_unknown_keys(
            baseline_section,
            {
                "model_name",
                "max_length",
                "extraction_batch_size",
                "training_batch_size",
                "hidden_size",
                "dropout",
                "epochs",
                "learning_rate",
                "weight_decay",
                "random_state",
                "dataloader_workers",
            },
            "baseline",
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

        device = _require_non_empty_string(gpu_section.get("device"), "gpu.device")
        if not device.startswith("cuda:") or not device.removeprefix("cuda:").isdigit():
            raise ConfigurationError(
                "gpu.device must identify one CUDA device, for example 'cuda:0'."
            )

        precision = _require_choice(
            gpu_section.get("precision"),
            "gpu.precision",
            {"bf16", "fp16", "fp32"},
        )
        attention_implementation = _require_choice(
            gpu_section.get("attention_implementation"),
            "gpu.attention_implementation",
            {"sdpa", "eager"},
        )
        allow_tf32 = _require_boolean(gpu_section.get("allow_tf32"), "gpu.allow_tf32")

        model_name = _require_non_empty_string(
            baseline_section.get("model_name"), "baseline.model_name"
        )
        max_length = _require_positive_integer(
            baseline_section.get("max_length"), "baseline.max_length"
        )
        extraction_batch_size = _require_positive_integer(
            baseline_section.get("extraction_batch_size"),
            "baseline.extraction_batch_size",
        )
        training_batch_size = _require_positive_integer(
            baseline_section.get("training_batch_size"),
            "baseline.training_batch_size",
        )
        hidden_size = _require_positive_integer(
            baseline_section.get("hidden_size"), "baseline.hidden_size"
        )
        dropout = _require_float(baseline_section.get("dropout"), "baseline.dropout")
        if not 0.0 <= dropout < 1.0:
            raise ConfigurationError("baseline.dropout must be in [0, 1).")

        epochs = _require_positive_integer(
            baseline_section.get("epochs"), "baseline.epochs"
        )
        learning_rate = _require_float(
            baseline_section.get("learning_rate"), "baseline.learning_rate"
        )
        if learning_rate <= 0.0:
            raise ConfigurationError("baseline.learning_rate must be greater than 0.")

        weight_decay = _require_float(
            baseline_section.get("weight_decay"), "baseline.weight_decay"
        )
        if weight_decay < 0.0:
            raise ConfigurationError("baseline.weight_decay cannot be negative.")

        baseline_random_state = _require_integer(
            baseline_section.get("random_state"), "baseline.random_state"
        )
        dataloader_workers = _require_integer(
            baseline_section.get("dataloader_workers"),
            "baseline.dataloader_workers",
        )
        if dataloader_workers < 0:
            raise ConfigurationError("baseline.dataloader_workers cannot be negative.")

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
            gpu=GPUConfig(
                device=device,
                precision=precision,
                attention_implementation=attention_implementation,
                allow_tf32=allow_tf32,
            ),
            baseline=BaselineConfig(
                model_name=model_name,
                max_length=max_length,
                extraction_batch_size=extraction_batch_size,
                training_batch_size=training_batch_size,
                hidden_size=hidden_size,
                dropout=dropout,
                epochs=epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                random_state=baseline_random_state,
                dataloader_workers=dataloader_workers,
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


def _require_positive_integer(value: Any, field_name: str) -> int:
    integer = _require_integer(value, field_name)
    if integer <= 0:
        raise ConfigurationError(f"{field_name} must be greater than 0.")
    return integer


def _require_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{field_name} must be a number.")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigurationError(f"{field_name} must be finite.")
    return number


def _require_boolean(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"{field_name} must be a boolean.")
    return value


def _require_choice(value: Any, field_name: str, choices: set[str]) -> str:
    choice = _require_non_empty_string(value, field_name).lower()
    if choice not in choices:
        allowed = ", ".join(sorted(choices))
        raise ConfigurationError(f"{field_name} must be one of: {allowed}.")
    return choice


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
