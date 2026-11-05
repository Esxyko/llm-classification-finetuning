"""Format elapsed durations consistently across metrics and command summaries."""

from __future__ import annotations


def format_duration(seconds: float, *, milliseconds: bool = False) -> str:
	"""Format unbounded hours, optionally rounding to millisecond precision."""
	if milliseconds:
		whole_seconds, millisecond_part = divmod(int(seconds * 1000 + 0.5), 1000)
	else:
		whole_seconds = int(seconds + 0.5)
	minutes, seconds = divmod(whole_seconds, 60)
	hours, minutes = divmod(minutes, 60)
	duration = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
	return f"{duration}.{millisecond_part:03d}" if milliseconds else duration
