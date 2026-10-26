"""Route Hugging Face downloads to a persistent native Hub cache."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

from .timing import format_duration


def load_pretrained(loader: Callable[..., Any], model_name: str, **kwargs: Any) -> Any:
	"""Apply MODEL_PATH to Hub downloads; retain standard behavior otherwise."""
	model_path = os.environ.get("MODEL_PATH", "").strip()
	if model_path and not Path(model_name).expanduser().is_dir():
		kwargs["cache_dir"] = str(Path(model_path).expanduser().resolve())
	return loader(model_name, **kwargs)


def load_model_weights(
	loader: Callable[..., Any], model_name: str, **kwargs: Any
) -> Any:
	"""Load weights with start/end logs instead of Hugging Face progress bars."""
	from huggingface_hub.utils import (
		are_progress_bars_disabled,
		disable_progress_bars,
		enable_progress_bars,
	)
	from transformers.utils import logging

	progress_enabled = logging.is_progress_bar_enabled()
	hub_progress_disabled = are_progress_bars_disabled()
	started = perf_counter()
	print(f"Loading weights: {model_name}.", flush=True)
	logging.disable_progress_bar()
	try:
		model = load_pretrained(loader, model_name, **kwargs)
	finally:
		if progress_enabled:
			logging.enable_progress_bar()
		if hub_progress_disabled:
			disable_progress_bars()
		else:
			enable_progress_bars()
	print(
		f"Loaded weights: {model_name} "
		f"(time: {format_duration(perf_counter() - started)}).",
		flush=True,
	)
	return model
