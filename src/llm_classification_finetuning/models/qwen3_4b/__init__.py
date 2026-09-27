"""Qwen3-4B frozen-backbone model profile."""

from ..profile import ModelProfile

PROFILE = ModelProfile(
    selector="qwen3-4b",
    config_key="qwen3_4b",
    artifact_stem="qwen3_4b",
    result_prefix="qwen3-4b",
)

__all__ = ("PROFILE",)
