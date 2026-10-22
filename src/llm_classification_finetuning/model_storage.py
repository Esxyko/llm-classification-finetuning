"""Route Hugging Face downloads to a persistent native Hub cache."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any


def load_pretrained(loader: Callable[..., Any], model_name: str, **kwargs: Any) -> Any:
	"""Apply MODEL_PATH to Hub downloads; retain standard behavior otherwise."""
	model_path = os.environ.get("MODEL_PATH", "").strip()
	if model_path and not Path(model_name).expanduser().is_dir():
		kwargs["cache_dir"] = str(Path(model_path).expanduser().resolve())
	return loader(model_name, **kwargs)
