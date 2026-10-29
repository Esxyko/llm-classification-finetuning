"""Frozen Skywork-Reward-V2-Qwen3-4B and its SR2 comparison head."""

from .profile import PROFILE


def __getattr__(name: str):
	"""Defer training imports until execution is requested."""
	if name != "SR2ModelPipeline":
		raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
	from .pipeline import SR2ModelPipeline

	globals()[name] = SR2ModelPipeline
	return SR2ModelPipeline


__all__ = ("PROFILE", "SR2ModelPipeline")
