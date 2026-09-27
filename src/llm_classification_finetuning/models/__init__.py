"""Selectable frozen-backbone pairwise model profiles."""

from .profile import ModelProfile
from .qwen3_1_7b import PROFILE as QWEN3_1_7B_PROFILE
from .qwen3_4b import PROFILE as QWEN3_4B_PROFILE

MODEL_PROFILES = {
    profile.selector: profile
    for profile in (QWEN3_1_7B_PROFILE, QWEN3_4B_PROFILE)
}

__all__ = (
    "MODEL_PROFILES",
    "ModelProfile",
    "QWEN3_1_7B_PROFILE",
    "QWEN3_4B_PROFILE",
)
