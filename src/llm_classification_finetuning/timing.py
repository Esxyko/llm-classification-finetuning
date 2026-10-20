"""Format elapsed durations consistently across metrics and command summaries."""

from __future__ import annotations


def format_duration(seconds: float) -> str:
	"""Round to whole seconds and format unbounded hours as HH:MM:SS."""
	minutes, seconds = divmod(int(seconds + 0.5), 60)
	hours, minutes = divmod(minutes, 60)
	return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
