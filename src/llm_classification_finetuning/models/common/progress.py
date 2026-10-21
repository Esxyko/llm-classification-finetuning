"""Report weighted training and validation progress at five-percent milestones."""

from __future__ import annotations

from time import perf_counter

from ...timing import format_duration


class TrainingProgress:
	"""Count weighted work across stages and epochs without GPU reads."""

	def __init__(
		self,
		*,
		model: str,
		mode: str,
		total_steps: float,
	) -> None:
		if total_steps <= 0:
			raise ValueError("Training progress requires a positive total work count.")
		self._model = model
		self._mode = mode
		self._total_steps = total_steps
		self._completed_steps = 0.0
		self._next_milestone = 1
		self._started = perf_counter()

	def advance(self, description: str, *, stage: str, weight: float = 1.0) -> None:
		"""Record weighted work and print once for any crossed milestones."""
		if self._completed_steps == self._total_steps:
			return
		self._completed_steps += weight
		milestone = int(self._completed_steps * 20 // self._total_steps)
		if milestone < self._next_milestone:
			return
		self._next_milestone = milestone + 1
		percentage = self._completed_steps * 100 / self._total_steps
		elapsed_seconds = perf_counter() - self._started
		estimated_seconds = elapsed_seconds * self._total_steps / self._completed_steps
		print(
			f"{self._model} {self._mode} {stage} {description}:\n"
			f"progress: {percentage:.1f}% "
			f"work: {self._completed_steps:,.1f}/{self._total_steps:,.1f} "
			f"time: {format_duration(elapsed_seconds)}/{format_duration(estimated_seconds)}",
			flush=True,
		)
