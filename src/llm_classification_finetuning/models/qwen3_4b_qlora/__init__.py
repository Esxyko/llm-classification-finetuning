"""Qwen3-4B QLoRA registration using the shared adapter training pipeline."""

from ..profile import ModelProfile
from ..qwen3_4b_lora import LoRAModelPipeline, LoRATrainingBatchRepository

PROFILE = ModelProfile(
	selector="qwen3-4b-qlora",
	config_key="qwen3_4b_qlora",
	artifact_stem="qwen3_4b_qlora",
	result_prefix="qwen3-4b-qlora",
	quantize_training=True,
)

__all__ = ("PROFILE", "LoRAModelPipeline", "LoRATrainingBatchRepository")
