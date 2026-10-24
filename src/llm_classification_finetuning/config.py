"""Load and validate application configuration."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .errors import ConfigurationError

MAX_VALIDATION_ROW_DEVIATION = 0.005

MAX_VALIDATION_ROW_DEVIATION = 0.005

LORA_WARMUP_FIELDS = (
	"warmup_epochs",
	"warmup_training_batch_size",
	"warmup_learning_rate",
)


@dataclass(frozen=True, slots=True)
class DataConfig:
	"""Locations and competition identity used by the data pipeline."""

	competition: str
	raw_dir: Path
	processed_path: Path


@dataclass(frozen=True, slots=True)
class ValidationConfig:
	"""Parameters controlling a deterministic grouped holdout split."""

	validation_fraction: float
	random_state: int
	tokenizer_name: str = "Qwen/Qwen3-4B"
	tokenizer_revision: str | None = None
	max_token_deviation: float = 0.005


@dataclass(frozen=True, slots=True)
class GPUConfig:
	"""Single-GPU training and two-GPU submission settings."""

	device: str
	precision: str
	attention_implementation: str
	allow_tf32: bool
	submit_devices: tuple[str, str] = ("cuda:0", "cuda:1")


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
	ab_swap: str


@dataclass(frozen=True, slots=True)
class WarmupModelConfig:
	"""Shared model, head, and optimization settings for head warmup."""

	model_name: str
	max_length: int
	hidden_size: int
	dropout: float
	weight_decay: float
	warmup_epochs: int
	extraction_batch_size: int
	warmup_training_batch_size: int
	warmup_learning_rate: float
	max_grad_norm: float
	random_state: int
	dataloader_workers: int
	ab_swap: str


@dataclass(frozen=True, slots=True)
class LoRAModelConfig(WarmupModelConfig):
	"""Effective adapter settings, with warmup optimization resolved from warmup."""

	training_batch_size: int
	head_inference_batch_size: int
	gradient_accumulation_steps: int
	epochs: int
	learning_rate: float
	head_learning_rate: float
	lora_rank: int
	lora_alpha: int
	lora_dropout: float
	lora_target_modules: tuple[str, ...]
	gradient_checkpointing_budget: int = 0


@dataclass(frozen=True, slots=True)
class AppConfig:
	"""Validated application configuration."""

	data: DataConfig
	validation: ValidationConfig
	gpu: GPUConfig
	qwen3_4b: ModelConfig | None = None
	qwen3_4b_lora: LoRAModelConfig | None = None
	warmup: WarmupModelConfig | None = None

	@classmethod
	def load(cls, config_path: Path) -> AppConfig:
		"""Read YAML and .env, resolving data paths from the working directory."""
		working_dir = Path.cwd()
		load_dotenv(dotenv_path=working_dir / ".env", override=False)
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
		if "cross_validation" in root:
			raise ConfigurationError(
				"Legacy cross_validation settings are no longer supported. Replace "
				"them with validation (validation_fraction: 0.2, "
				"max_token_deviation: 0.005), remove n_splits, set data.processed_path "
				"to data/processed/train_validation.parquet, and run 'uv run preprocess'."
			)
		_reject_unknown_keys(
			root,
			{
				"data",
				"validation",
				"gpu",
				"qwen3_4b",
				"qwen3_4b_lora",
				"warmup",
			},
			"configuration root",
		)

		data_section = _require_mapping(root.get("data"), "data")
		_reject_unknown_keys(
			data_section,
			{"competition", "raw_dir", "processed_path"},
			"data",
		)
		validation_section = _require_mapping(root.get("validation"), "validation")
		_reject_unknown_keys(
			validation_section,
			{
				"validation_fraction",
				"random_state",
				"tokenizer_name",
				"tokenizer_revision",
				"max_token_deviation",
			},
			"validation",
		)
		gpu_section = _require_mapping(root.get("gpu"), "gpu")
		_reject_unknown_keys(
			gpu_section,
			{
				"device",
				"submit_devices",
				"precision",
				"attention_implementation",
				"allow_tf32",
			},
			"gpu",
		)
		qwen3_4b = _load_model_config(root, "qwen3_4b") if "qwen3_4b" in root else None
		warmup = _load_warmup_model_config(root) if "warmup" in root else None
		qwen3_4b_lora = (
			_load_lora_model_config(root, warmup) if "qwen3_4b_lora" in root else None
		)

		competition = _require_non_empty_string(
			data_section.get("competition"), "data.competition"
		)
		raw_dir = _resolve_data_path(
			working_dir,
			_require_non_empty_string(data_section.get("raw_dir"), "data.raw_dir"),
		)
		processed_path = _resolve_data_path(
			working_dir,
			_require_non_empty_string(
				data_section.get("processed_path"), "data.processed_path"
			),
		)
		if processed_path.suffix.lower() != ".parquet":
			raise ConfigurationError(
				"data.processed_path must point to a .parquet file."
			)

		validation_fraction = _require_float(
			validation_section.get("validation_fraction", 0.2),
			"validation.validation_fraction",
		)
		if not 0.0 < validation_fraction < 1.0:
			raise ConfigurationError(
				"validation.validation_fraction must be in (0, 1)."
			)

		random_state = _require_integer(
			validation_section.get("random_state"),
			"validation.random_state",
		)
		tokenizer_name = _require_non_empty_string(
			validation_section.get("tokenizer_name", "Qwen/Qwen3-4B"),
			"validation.tokenizer_name",
		)
		tokenizer_revision = validation_section.get("tokenizer_revision")
		if tokenizer_revision is not None:
			tokenizer_revision = _require_non_empty_string(
				tokenizer_revision, "validation.tokenizer_revision"
			)
		max_token_deviation = _require_float(
			validation_section.get("max_token_deviation", 0.005),
			"validation.max_token_deviation",
		)
		if not 0.0 < max_token_deviation < 1.0:
			raise ConfigurationError(
				"validation.max_token_deviation must be in (0, 1)."
			)

		device = _require_cuda_device(gpu_section.get("device"), "gpu.device")
		submit_devices = gpu_section.get("submit_devices", ["cuda:0", "cuda:1"])
		if not isinstance(submit_devices, list) or len(submit_devices) != 2:
			raise ConfigurationError(
				"gpu.submit_devices must be a list of exactly two CUDA devices."
			)
		first_submit_device, second_submit_device = (
			_require_cuda_device(value, f"gpu.submit_devices[{index}]")
			for index, value in enumerate(submit_devices)
		)
		if first_submit_device == second_submit_device:
			raise ConfigurationError(
				"gpu.submit_devices must contain distinct devices."
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

		return cls(
			data=DataConfig(
				competition=competition,
				raw_dir=raw_dir,
				processed_path=processed_path,
			),
			validation=ValidationConfig(
				validation_fraction=validation_fraction,
				random_state=random_state,
				tokenizer_name=tokenizer_name,
				tokenizer_revision=tokenizer_revision,
				max_token_deviation=max_token_deviation,
			),
			gpu=GPUConfig(
				device=device,
				precision=precision,
				attention_implementation=attention_implementation,
				allow_tf32=allow_tf32,
				submit_devices=(first_submit_device, second_submit_device),
			),
			qwen3_4b=qwen3_4b,
			qwen3_4b_lora=qwen3_4b_lora,
			warmup=warmup,
		)


def _load_warmup_model_config(root: Mapping[str, Any]) -> WarmupModelConfig:
	section = _require_mapping(root.get("warmup"), "warmup")
	keys = set(WarmupModelConfig.__dataclass_fields__) - {"ab_swap"}
	_reject_unknown_keys(section, keys | {"A/B_swap"}, "warmup")
	return _load_warmup_fields(section, "warmup")


def _load_warmup_fields(
	section: Mapping[str, Any], section_name: str
) -> WarmupModelConfig:
	"""Validate the same head-warmup fields for standalone and adapter training."""

	def field(name: str) -> str:
		return f"{section_name}.{name}"

	def positive_integer(name: str) -> int:
		return _require_positive_integer(section.get(name), field(name))

	def positive_float(name: str) -> float:
		value = _require_float(section.get(name), field(name))
		if value <= 0.0:
			raise ConfigurationError(f"{field(name)} must be greater than 0.")
		return value

	dropout = _require_float(section.get("dropout"), field("dropout"))
	if not 0.0 <= dropout < 1.0:
		raise ConfigurationError(f"{field('dropout')} must be in [0, 1).")
	weight_decay = _require_float(section.get("weight_decay"), field("weight_decay"))
	if weight_decay < 0.0:
		raise ConfigurationError(f"{field('weight_decay')} cannot be negative.")
	warmup_epochs = _require_integer(
		section.get("warmup_epochs"), field("warmup_epochs")
	)
	if warmup_epochs < 0:
		raise ConfigurationError(f"{field('warmup_epochs')} cannot be negative.")
	workers = _require_integer(
		section.get("dataloader_workers"), field("dataloader_workers")
	)
	if workers < 0:
		raise ConfigurationError(f"{field('dataloader_workers')} cannot be negative.")
	return WarmupModelConfig(
		model_name=_require_non_empty_string(
			section.get("model_name"), field("model_name")
		),
		max_length=positive_integer("max_length"),
		hidden_size=positive_integer("hidden_size"),
		dropout=dropout,
		weight_decay=weight_decay,
		warmup_epochs=warmup_epochs,
		extraction_batch_size=positive_integer("extraction_batch_size"),
		warmup_training_batch_size=positive_integer("warmup_training_batch_size"),
		warmup_learning_rate=positive_float("warmup_learning_rate"),
		max_grad_norm=positive_float("max_grad_norm"),
		random_state=_require_integer(
			section.get("random_state"), field("random_state")
		),
		dataloader_workers=workers,
		ab_swap=_require_choice(
			section.get("A/B_swap", "aug"), field("A/B_swap"), {"aug", "inf"}
		),
	)


def _load_lora_model_config(
	root: Mapping[str, Any],
	warmup: WarmupModelConfig | None,
	section_name: str = "qwen3_4b_lora",
) -> LoRAModelConfig:
	section = _require_mapping(root.get(section_name), section_name)
	keys = set(LoRAModelConfig.__dataclass_fields__) - {"ab_swap", *LORA_WARMUP_FIELDS}
	_reject_unknown_keys(section, keys | {"A/B_swap"}, section_name)
	if warmup is None:
		raise ConfigurationError(
			f"{section_name} requires a 'warmup' section. Move "
			"warmup_epochs, warmup_training_batch_size, and warmup_learning_rate "
			"there; set warmup.warmup_epochs to 0 to disable head warmup."
		)

	def field(name: str) -> str:
		return f"{section_name}.{name}"

	def positive_integer(name: str) -> int:
		return _require_positive_integer(section.get(name), field(name))

	def positive_float(name: str) -> float:
		value = _require_float(section.get(name), field(name))
		if value <= 0.0:
			raise ConfigurationError(f"{field(name)} must be greater than 0.")
		return value

	def fraction(name: str) -> float:
		value = _require_float(section.get(name), field(name))
		if not 0.0 <= value < 1.0:
			raise ConfigurationError(f"{field(name)} must be in [0, 1).")
		return value

	shared = _load_warmup_fields(
		{**section, **{name: getattr(warmup, name) for name in LORA_WARMUP_FIELDS}},
		section_name,
	)
	token_budget = _require_integer(
		section.get("gradient_checkpointing_budget", 0),
		field("gradient_checkpointing_budget"),
	)
	if token_budget < 0:
		raise ConfigurationError(
			f"{field('gradient_checkpointing_budget')} cannot be negative."
		)
	targets = section.get("lora_target_modules")
	if not isinstance(targets, list) or not targets:
		raise ConfigurationError(
			f"{field('lora_target_modules')} must be a non-empty list."
		)
	target_modules = tuple(
		_require_non_empty_string(value, field(f"lora_target_modules[{index}]"))
		for index, value in enumerate(targets)
	)
	if len(set(target_modules)) != len(target_modules):
		raise ConfigurationError(
			f"{field('lora_target_modules')} cannot contain duplicates."
		)

	return LoRAModelConfig(
		**asdict(shared),
		training_batch_size=positive_integer("training_batch_size"),
		head_inference_batch_size=positive_integer("head_inference_batch_size"),
		gradient_accumulation_steps=positive_integer("gradient_accumulation_steps"),
		epochs=positive_integer("epochs"),
		learning_rate=positive_float("learning_rate"),
		head_learning_rate=positive_float("head_learning_rate"),
		lora_rank=positive_integer("lora_rank"),
		lora_alpha=positive_integer("lora_alpha"),
		lora_dropout=fraction("lora_dropout"),
		lora_target_modules=target_modules,
		gradient_checkpointing_budget=token_budget,
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
		"A/B_swap",
	}
	_reject_unknown_keys(section, allowed_keys, section_name)

	def field(name: str) -> str:
		return f"{section_name}.{name}"

	dropout = _require_float(section.get("dropout"), field("dropout"))
	if not 0.0 <= dropout < 1.0:
		raise ConfigurationError(f"{field('dropout')} must be in [0, 1).")
	learning_rate = _require_float(section.get("learning_rate"), field("learning_rate"))
	if learning_rate <= 0.0:
		raise ConfigurationError(f"{field('learning_rate')} must be greater than 0.")
	weight_decay = _require_float(section.get("weight_decay"), field("weight_decay"))
	if weight_decay < 0.0:
		raise ConfigurationError(f"{field('weight_decay')} cannot be negative.")
	dataloader_workers = _require_integer(
		section.get("dataloader_workers"), field("dataloader_workers")
	)
	if dataloader_workers < 0:
		raise ConfigurationError(f"{field('dataloader_workers')} cannot be negative.")

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
		ab_swap=_require_choice(
			section.get("A/B_swap", "aug"), field("A/B_swap"), {"aug", "inf"}
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


def _require_cuda_device(value: Any, field_name: str) -> str:
	device = _require_non_empty_string(value, field_name)
	index = device.removeprefix("cuda:")
	if not device.startswith("cuda:") or not index.isascii() or not index.isdigit():
		raise ConfigurationError(
			f"{field_name} must identify one CUDA device, for example 'cuda:0'."
		)
	return f"cuda:{int(index)}"


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


def _resolve_data_path(base_dir: Path, configured_path: str) -> Path:
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
