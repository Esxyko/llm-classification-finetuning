"""Publish model validation runs and test submissions."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ...config import LORA_WARMUP_FIELDS, AppConfig, LoRAModelConfig
from ...errors import ModelExecutionError
from ...timing import format_duration
from ..profile import ModelProfile
from .hardware import GPUMemoryPeak
from .model import PairwiseClassificationHead
from .trainer import ValidationTrainingResult


class ModelResultPublisher:
	"""Atomically publish one completed holdout validation run."""

	PROBABILITY_COLUMNS = ("winner_model_a", "winner_model_b", "winner_tie")

	def __init__(self, results_root: Path, result_prefix: str) -> None:
		self._results_root = results_root.resolve()
		self._result_prefix = result_prefix

	def publish(
		self,
		result: ValidationTrainingResult,
		config: AppConfig,
		profile: ModelProfile,
		*,
		prediction_aggregation: str | None = None,
	) -> tuple[Path, float, int]:
		"""Publish validation probabilities, metrics, and the used config together."""
		model_config = profile.resolve_config(config)
		generated_at = datetime.now(UTC).astimezone()
		run_name = generated_at.strftime(f"{self._result_prefix}-%Y%m%d-%H%M%S")
		output_dir = self._results_root / run_name
		if output_dir.exists():
			raise ModelExecutionError(
				f"Model result directory already exists for this second: {output_dir}"
			)
		divisor = 2 if prediction_aggregation == "ab_swap_average" else 1
		if not len(result.ids) or len(result.ids) % divisor:
			raise ModelExecutionError(
				"Validation predictions must be nonempty, with complete orientation pairs when averaging."
			)
		prediction_count = len(result.ids) // divisor
		metrics = {
			"run_name": run_name,
			"generated_at": generated_at.isoformat(timespec="seconds"),
			"model_name": model_config.model_name,
			"head_architecture": profile.head_architecture,
			"embedding_combination": "sum_signed_diff_abs_diff"
			if profile.head_architecture == "swap_equivariant_reparam_v2"
			else PairwiseClassificationHead.embedding_combination(),
			"validation_strategy": "holdout",
			"training_rows": result.training_rows,
			"source_training_rows": result.source_training_rows,
			"validation_rows": prediction_count,
			"validation_fraction": prediction_count
			/ (result.source_training_rows + prediction_count),
			"predictions": prediction_count,
			"validation_loss": result.final_loss,
			"training_time": format_duration(result.training_time_seconds),
			"validation_time": format_duration(result.validation_time_seconds),
			"total_time": format_duration(result.total_time_seconds),
			"epochs": [
				{
					"epoch": epoch.epoch,
					"training_loss": epoch.training_loss,
					"validation_loss": epoch.validation_loss,
					"training_time": format_duration(epoch.training_time_seconds),
					"validation_time": format_duration(epoch.validation_time_seconds),
					"total_time": format_duration(epoch.total_time_seconds),
				}
				for epoch in result.epochs
			],
		}
		if prediction_aggregation is not None:
			metrics["prediction_aggregation"] = prediction_aggregation
		try:
			self._results_root.mkdir(parents=True, exist_ok=True)
			with tempfile.TemporaryDirectory(
				dir=self._results_root, prefix=f".{self._result_prefix}-"
			) as temporary_dir:
				staging_dir = Path(temporary_dir)
				frame = pd.DataFrame(
					{
						"id": result.ids,
						**{
							column: result.probabilities[:, index]
							for index, column in enumerate(self.PROBABILITY_COLUMNS)
						},
					}
				)
				self._validate_probabilities(frame)
				frame.to_csv(staging_dir / "validation.csv", index=False)
				(staging_dir / "metrics.json").write_text(
					json.dumps(metrics, indent=2, sort_keys=True) + "\n",
					encoding="utf-8",
				)
				(staging_dir / "config.yaml").write_text(
					yaml.safe_dump(
						self._configuration_snapshot(config, profile),
						sort_keys=False,
						allow_unicode=True,
					),
					encoding="utf-8",
				)
				os.replace(staging_dir, output_dir)
		except ModelExecutionError:
			raise
		except Exception as error:
			raise ModelExecutionError(
				f"Could not publish model results to {output_dir}: {error}"
			) from error
		print(f"Wrote validation results: {output_dir}", flush=True)
		return output_dir, result.final_loss, prediction_count

	@staticmethod
	def record_model_time(
		output_dir: Path,
		elapsed_seconds: float,
		*,
		gpu_memory: GPUMemoryPeak | None = None,
	) -> None:
		"""Atomically record complete duration and optional GPU peaks for this model."""
		metrics_path = output_dir / "metrics.json"
		try:
			metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
			metrics["total_time"] = format_duration(elapsed_seconds)
			metrics.pop("config", None)
			metrics.pop("config_total_time", None)
			metrics["config_total_time"] = format_duration(elapsed_seconds)
			if gpu_memory is not None:
				metrics["gpu_memory"] = asdict(gpu_memory)
			with tempfile.TemporaryDirectory(
				dir=output_dir, prefix=".metrics-"
			) as temporary_dir:
				staged_metrics = Path(temporary_dir) / "metrics.json"
				staged_metrics.write_text(
					json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
				)
				os.replace(staged_metrics, metrics_path)
		except Exception as error:
			raise ModelExecutionError(
				f"Could not record model runtime metrics in {metrics_path}: {error}"
			) from error

	@staticmethod
	def _configuration_snapshot(
		config: AppConfig, profile: ModelProfile
	) -> dict[str, dict[str, object]]:
		"""Serialize current-model settings with its required shared warmup section."""
		model_config = profile.resolve_config(config)
		sections = {
			"data": asdict(config.data),
			"validation": asdict(config.validation),
			"gpu": asdict(config.gpu),
		}
		model_section = asdict(model_config)
		if isinstance(model_config, LoRAModelConfig):
			if config.warmup is None:
				raise ModelExecutionError("LoRA results require a 'warmup' section.")
			sections["warmup"] = asdict(config.warmup)
			for key in (*LORA_WARMUP_FIELDS, "warmup_extraction_budget"):
				model_section.pop(key)
			if model_config.ab_swap == "none":
				model_section.pop("ab_swap")
				sections["warmup"].pop("ab_swap")
		sections[profile.config_key] = model_section
		return {
			section_name: {
				"A/B_swap" if key == "ab_swap" else key: (
					str(value)
					if isinstance(value, Path)
					else list(value)
					if isinstance(value, tuple)
					else value
				)
				for key, value in section.items()
			}
			for section_name, section in sections.items()
		}

	def _validate_probabilities(self, predictions: pd.DataFrame) -> None:
		values = predictions.loc[:, list(self.PROBABILITY_COLUMNS)].to_numpy(
			dtype=np.float64
		)
		if not _probabilities_are_valid(values):
			raise ModelExecutionError("The holdout run produced invalid probabilities.")


class ModelSubmissionPublisher:
	"""Atomically publish one timestamped competition submission."""

	PROBABILITY_COLUMNS = ModelResultPublisher.PROBABILITY_COLUMNS

	def __init__(self, results_root: Path, result_prefix: str) -> None:
		self._results_root = results_root.resolve()
		self._result_prefix = result_prefix

	def publish(
		self,
		ids: tuple[int, ...],
		probabilities: np.ndarray,
	) -> tuple[Path, Path]:
		"""Write a submission CSV in a newly exposed submission-run directory."""
		if probabilities.shape != (len(ids), len(self.PROBABILITY_COLUMNS)):
			raise ModelExecutionError(
				"Test probabilities do not match the test ID count."
			)
		if not _probabilities_are_valid(probabilities):
			raise ModelExecutionError(
				"Model submission inference produced invalid probabilities."
			)

		generated_at = datetime.now(UTC).astimezone()
		output_dir = self._results_root / generated_at.strftime(
			f"{self._result_prefix}-%Y%m%d-%H%M%S"
		)
		submission_path = output_dir / "submission.csv"
		try:
			self._results_root.mkdir(parents=True, exist_ok=True)
			if output_dir.exists():
				raise ModelExecutionError(
					"Model submission result directory already exists for this second: "
					f"{output_dir}"
				)
			with tempfile.TemporaryDirectory(
				dir=self._results_root,
				prefix=f".{self._result_prefix}-",
			) as temporary_dir:
				staging_dir = Path(temporary_dir)
				pd.DataFrame(
					{
						"id": ids,
						**{
							column: probabilities[:, index]
							for index, column in enumerate(self.PROBABILITY_COLUMNS)
						},
					}
				).to_csv(staging_dir / "submission.csv", index=False)
				os.replace(staging_dir, output_dir)
		except ModelExecutionError:
			raise
		except Exception as error:
			raise ModelExecutionError(
				f"Could not publish model submission results to {output_dir}: {error}"
			) from error
		return output_dir, submission_path


def _probabilities_are_valid(values: np.ndarray) -> bool:
	return bool(
		values.size > 0
		and np.isfinite(values).all()
		and not (values < 0.0).any()
		and not (values > 1.0).any()
		and np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-6)
	)
