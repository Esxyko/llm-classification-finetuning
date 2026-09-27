"""Selectable frozen-backbone comparison model profiles and pipelines."""

from .common import ModelPipeline
from .profile import ModelProfile
from .qwen3_1_7b import PROFILE as QWEN3_1_7B_PROFILE
from .qwen3_1_7b_mono_input import PROFILE as QWEN3_1_7B_MONO_INPUT_PROFILE
from .qwen3_1_7b_mono_input import MonoInputModelPipeline
from .qwen3_4b import PROFILE as QWEN3_4B_PROFILE

MODEL_PROFILES = {
    profile.selector: profile
    for profile in (
        QWEN3_1_7B_PROFILE,
        QWEN3_1_7B_MONO_INPUT_PROFILE,
        QWEN3_4B_PROFILE,
    )
}
MODEL_PIPELINES = {
    QWEN3_1_7B_PROFILE.selector: ModelPipeline,
    QWEN3_1_7B_MONO_INPUT_PROFILE.selector: MonoInputModelPipeline,
    QWEN3_4B_PROFILE.selector: ModelPipeline,
}

__all__ = (
    "MODEL_PIPELINES",
    "MODEL_PROFILES",
    "QWEN3_1_7B_MONO_INPUT_PROFILE",
    "QWEN3_1_7B_PROFILE",
    "QWEN3_4B_PROFILE",
    "ModelProfile",
)
