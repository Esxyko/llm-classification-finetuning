"""Skywork-Reward-V2-Qwen3-4B LoRA profile with the EXPT2 comparison head."""

from ..profile import ModelProfile

PROFILE = ModelProfile(
	selector="SR2-Qwen3-4B-lora",
	config_key="SR2_Qwen3_4B_lora",
	artifact_stem="SR2_Qwen3_4B_lora",
	result_prefix="SR2-Qwen3-4B-lora",
	head_architecture="swap_equivariant_tie_from_wins_v1",
	ab_swap_override="none",
)

__all__ = ("PROFILE",)
