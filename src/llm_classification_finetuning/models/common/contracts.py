"""Execution interfaces and summaries shared by frozen and trainable models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from ...config import AppConfig
from ..profile import ModelProfile


class ModelMode(Enum):
	"""Supported model command execution modes."""

	VALIDATION = "validation"
	BUILD = "build"
	SUBMIT = "submit"


@dataclass(frozen=True, slots=True)
class ModelRunResult:
	"""Summary of a completed single-holdout model run."""

	output_dir: Path
	predictions: int
	validation_loss: float
	cache_path: Path | None = None
	cache_reused: bool = False


@dataclass(frozen=True, slots=True)
class ModelBuildResult:
	"""Summary of a completed full-data model build."""

	checkpoint_path: Path
	training_rows: int
	epochs: int
	final_training_loss: float
	cache_path: Path | None = None
	cache_reused: bool = False
	executed_warmup_epochs: int | None = None
	executed_joint_epochs: int | None = None
	merged_backbone_path: Path | None = None


@dataclass(frozen=True, slots=True)
class ModelSubmitResult:
	"""Summary of a completed competition submission run."""

	output_dir: Path
	submission_path: Path
	predictions: int
	cache_path: Path | None = None
	cache_reused: bool = False


@dataclass(frozen=True, slots=True)
class WarmupBuildResult:
	"""Summary and current train/validation losses of a head-warmup build or hit."""

	cache_path: Path
	cache_reused: bool
	training_rows: int
	epochs: int
	final_training_loss: float
	validation_loss: float


ModelExecutionResult = (
	ModelRunResult | ModelBuildResult | ModelSubmitResult | WarmupBuildResult
)


class ModelPipelineInterface(Protocol):
	"""Constructor and execution contract for registered model pipelines."""

	def __init__(
		self,
		config: AppConfig,
		project_root: Path,
		profile: ModelProfile,
		checkpoint_tag: str | None = None,
	) -> None: ...

	def run(
		self,
		*,
		mode: ModelMode = ModelMode.VALIDATION,
		refresh_cache: bool = False,
	) -> ModelExecutionResult: ...
