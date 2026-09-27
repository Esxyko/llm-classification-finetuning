"""Qwen3-1.7B structured mono-input model definition."""

from ..profile import ModelProfile
from .model import INPUT_INSTRUCTION
from .pipeline import MonoInputModelPipeline

PROFILE = ModelProfile(
    selector="qwen3-1.7b-mono-input",
    config_key="qwen3_1_7b_mono_input",
    artifact_stem="qwen3_1_7b_mono_input",
    result_prefix="qwen3-1.7b-mono-input",
)

__all__ = ("INPUT_INSTRUCTION", "PROFILE", "MonoInputModelPipeline")
