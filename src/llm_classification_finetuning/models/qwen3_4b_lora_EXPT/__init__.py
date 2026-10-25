"""Qwen3-4B LoRA profile with a swap-equivariant classification head."""

from ..profile import ModelProfile

PROFILE = ModelProfile(
	selector="qwen3-4b-lora-EXPT",
	config_key="qwen3_4b_lora_EXPT",
	artifact_stem="qwen3_4b_lora_EXPT",
	result_prefix="qwen3-4b-lora-EXPT",
	head_architecture="swap_equivariant_reparam_v1",
)

__all__ = ("PROFILE",)
