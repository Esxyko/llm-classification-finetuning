"""Route Hugging Face downloads to a persistent native Hub cache."""

from __future__ import annotations

import os
from collections.abc import Callable
from logging import Filter, LogRecord
from pathlib import Path
from time import perf_counter
from typing import Any

from .errors import ModelExecutionError
from .timing import format_duration


class _LoadReportFilter(Filter):
	"""Suppress Transformers weight-loading report records."""

	def filter(self, record: LogRecord) -> bool:
		return " LOAD REPORT" not in record.getMessage()


def load_pretrained(loader: Callable[..., Any], model_name: str, **kwargs: Any) -> Any:
	"""Route Hub storage and explain missing local files during offline execution."""
	from huggingface_hub.constants import HF_HUB_OFFLINE

	model_path = os.environ.get("MODEL_PATH", "").strip()
	if model_path and not Path(model_name).expanduser().is_dir():
		kwargs["cache_dir"] = str(Path(model_path).expanduser().resolve())
	if HF_HUB_OFFLINE:
		kwargs["local_files_only"] = True
	try:
		return loader(model_name, **kwargs)
	except OSError as error:
		if not (HF_HUB_OFFLINE or kwargs.get("local_files_only")):
			raise
		from huggingface_hub.constants import HF_HUB_CACHE

		location = (
			str(Path(model_name).expanduser().resolve())
			if Path(model_name).expanduser().is_dir()
			else str(kwargs.get("cache_dir") or HF_HUB_CACHE)
		)
		raise ModelExecutionError(
			f"Offline Hugging Face load failed for {model_name!r} "
			f"(revision={kwargs.get('revision') or 'main'}, location={location}). "
			"Required local files are missing or unreadable. Restore the complete "
			"config/tokenizer/weight cache, including all referenced shards and blobs, "
			"at MODEL_PATH (or the configured local model directory). "
			"Download missing files separately with HF_HUB_OFFLINE=0 before rerunning. "
			f"Original error: {error}"
		) from error


def load_model_weights(
	loader: Callable[..., Any], model_name: str, **kwargs: Any
) -> Any:
	"""Load weights with start/end logs, hiding progress bars and load reports."""
	from huggingface_hub.utils import (
		are_progress_bars_disabled,
		disable_progress_bars,
		enable_progress_bars,
	)
	from transformers.utils import logging

	progress_enabled = logging.is_progress_bar_enabled()
	hub_progress_disabled = are_progress_bars_disabled()
	model_logger = logging.get_logger("transformers.modeling_utils")
	load_report_filter = _LoadReportFilter()
	started = perf_counter()
	print(f"Loading weights: {model_name}.", flush=True)
	logging.disable_progress_bar()
	model_logger.addFilter(load_report_filter)
	try:
		model = load_pretrained(loader, model_name, **kwargs)
	finally:
		model_logger.removeFilter(load_report_filter)
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
