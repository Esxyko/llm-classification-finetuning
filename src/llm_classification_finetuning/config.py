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
    """Project-wide settings for the configured CUDA devices."""

    devices: tuple[str, ...]
    precision: str
    attention_implementation: str
    allow_tf32: bool


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Hyperparameters for one frozen-backbone comparison model."""

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
    qwen3_1_7b: ModelConfig
    qwen3_1_7b_mono_input: ModelConfig
    qwen3_4b: ModelConfig

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
            {
                "data",
                "cross_validation",
                "gpu",
                "qwen3_1_7b",
                "qwen3_1_7b_mono_input",
                "qwen3_4b",
            },
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
            {"devices", "precision", "attention_implementation", "allow_tf32"},
            "gpu",
        )
        qwen3_1_7b = _load_model_config(root, "qwen3_1_7b")
        qwen3_1_7b_mono_input = _load_model_config(
            root,
            "qwen3_1_7b_mono_input",
        )
        qwen3_4b = _load_model_config(root, "qwen3_4b")

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

        devices_value = gpu_section.get("devices")
        if not isinstance(devices_value, list) or not devices_value:
            raise ConfigurationError(
                "gpu.devices must be a non-empty YAML list of CUDA devices."
            )
        devices = tuple(
            _require_non_empty_string(value, f"gpu.devices[{index}]")
            for index, value in enumerate(devices_value)
        )
        if any(
            not device.startswith("cuda:")
            or not device.removeprefix("cuda:").isdigit()
            for device in devices
        ):
            raise ConfigurationError(
                "Every gpu.devices entry must identify a CUDA device, for example "
                "'cuda:0'."
            )
        if len(set(devices)) != len(devices):
            raise ConfigurationError("gpu.devices cannot contain duplicates.")

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
                devices=devices,
                precision=precision,
                attention_implementation=attention_implementation,
                allow_tf32=allow_tf32,
            ),
            qwen3_1_7b=qwen3_1_7b,
            qwen3_1_7b_mono_input=qwen3_1_7b_mono_input,
            qwen3_4b=qwen3_4b,
        )


def _load_model_config(root: Mapping[str, Any], section_name: str) -> ModelConfig:
    section = _require_mapping(root.get(section_name), section_name)
    allowed_keys = {
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
    }
    _reject_unknown_keys(section, allowed_keys, section_name)

    def field(name: str) -> str:
        return f"{section_name}.{name}"

    dropout = _require_float(section.get("dropout"), field("dropout"))
    if not 0.0 <= dropout < 1.0:
        raise ConfigurationError(f"{field('dropout')} must be in [0, 1).")
    learning_rate = _require_float(
        section.get("learning_rate"), field("learning_rate")
    )
    if learning_rate <= 0.0:
        raise ConfigurationError(
            f"{field('learning_rate')} must be greater than 0."
        )
    weight_decay = _require_float(
        section.get("weight_decay"), field("weight_decay")
    )
    if weight_decay < 0.0:
        raise ConfigurationError(f"{field('weight_decay')} cannot be negative.")
    dataloader_workers = _require_integer(
        section.get("dataloader_workers"), field("dataloader_workers")
    )
    if dataloader_workers < 0:
        raise ConfigurationError(
            f"{field('dataloader_workers')} cannot be negative."
        )

    return ModelConfig(
        model_name=_require_non_empty_string(
            section.get("model_name"), field("model_name")
        ),
        max_length=_require_positive_integer(
            section.get("max_length"), field("max_length")
        ),
        extraction_batch_size=_require_positive_integer(
            section.get("extraction_batch_size"), field("extraction_batch_size")
        ),
        training_batch_size=_require_positive_integer(
            section.get("training_batch_size"), field("training_batch_size")
        ),
        hidden_size=_require_positive_integer(
            section.get("hidden_size"), field("hidden_size")
        ),
        dropout=dropout,
        epochs=_require_positive_integer(section.get("epochs"), field("epochs")),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        random_state=_require_integer(
            section.get("random_state"), field("random_state")
        ),
        dataloader_workers=dataloader_workers,
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
