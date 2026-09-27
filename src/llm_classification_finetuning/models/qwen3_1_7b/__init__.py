"""Qwen3-1.7B frozen-backbone model profile."""

from ..profile import ModelProfile

PROFILE = ModelProfile(
    selector="qwen3-1.7b",
    config_key="qwen3_1_7b",
    artifact_stem="qwen3_1_7b",
    result_prefix="qwen3-1.7b",
)

__all__ = ("PROFILE",)
