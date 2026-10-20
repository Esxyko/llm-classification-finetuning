"""Qwen3-4B pairwise LoRA model registration."""

from ..profile import ModelProfile
from .batching import (
	LoRATrainingBatchRepository,
	PreparedTrainingSplit,
	TrainingBatchPlan,
)
from .pipeline import LoRAModelPipeline

PROFILE = ModelProfile(
	selector="qwen3-4b-lora",
	config_key="qwen3_4b_lora",
	artifact_stem="qwen3_4b_lora",
	result_prefix="qwen3-4b-lora",
)

__all__ = (
	"PROFILE",
	"LoRAModelPipeline",
	"LoRATrainingBatchRepository",
	"PreparedTrainingSplit",
	"TrainingBatchPlan",
)
