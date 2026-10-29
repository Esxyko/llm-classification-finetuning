"""Frozen Skywork model identity, independent of the shared warmup namespace."""

from ..profile import ModelProfile

PROFILE = ModelProfile(
	selector="SR2-Qwen3-4B",
	config_key="SR2_Qwen3_4B",
	artifact_stem="SR2_Qwen3_4B",
	result_prefix="SR2-Qwen3-4B",
	head_architecture="swap_equivariant_harmonic_discrepancy_v3",
	ab_swap_override="none",
	supports_submit=False,
)

__all__ = ("PROFILE",)
