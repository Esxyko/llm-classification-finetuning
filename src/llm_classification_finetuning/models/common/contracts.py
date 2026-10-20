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

	CROSS_VALIDATION = "cross-validation"
	BUILD = "build"
	TEST = "test"


@dataclass(frozen=True, slots=True)
class ModelRunResult:
	"""Summary of a completed model cross-validation run."""

	output_dir: Path
	folds: int
	predictions: int
	average_loss: float
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


@dataclass(frozen=True, slots=True)
class ModelTestResult:
	"""Summary of a completed competition test inference run."""

	output_dir: Path
	submission_path: Path
	predictions: int
	cache_path: Path | None = None
	cache_reused: bool = False


ModelExecutionResult = ModelRunResult | ModelBuildResult | ModelTestResult


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
		mode: ModelMode = ModelMode.CROSS_VALIDATION,
		refresh_cache: bool = False,
	) -> ModelExecutionResult: ...
