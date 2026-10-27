"""Report training work with independent stage timers and progress milestones."""

from __future__ import annotations

from math import isfinite
from time import perf_counter

from ...timing import format_duration


class TrainingProgress:
	"""Count work without GPU reads, optionally resetting at stage boundaries."""

	def __init__(
		self,
		*,
		model: str,
		mode: str,
		total_steps: float | None = None,
	) -> None:
		self._model = model
		self._mode = mode
		self._total_steps = 0.0
		self._completed_steps = 0.0
		self._next_milestone = 1
		self._milestone_percent = 5
		self._started: float | None = None
		self._stage: str | None = None
		self._finished = False
		if total_steps is not None:
			self._reset(total_steps, stage=None, milestone_percent=5)

	def start_stage(
		self, *, stage: str, total_steps: float, milestone_percent: int = 20
	) -> None:
		"""Start a fresh stage whose final log is deferred until finish_stage()."""
		self._reset(total_steps, stage=stage, milestone_percent=milestone_percent)

	def _reset(
		self, total_steps: float, *, stage: str | None, milestone_percent: int
	) -> None:
		if not isfinite(total_steps) or total_steps <= 0:
			raise ValueError("Training progress requires a positive total work count.")
		if not 1 <= milestone_percent <= 100:
			raise ValueError("Progress milestones must be between 1 and 100 percent.")
		self._total_steps = total_steps
		self._completed_steps = 0.0
		self._next_milestone = 1
		self._milestone_percent = milestone_percent
		self._stage = stage
		self._finished = False
		self._started = perf_counter()

	def advance(self, description: str, *, stage: str, weight: float = 1.0) -> None:
		"""Record weighted work and print once for any crossed milestones."""
		if self._started is None:
			raise RuntimeError("Start a progress stage before recording work.")
		if self._finished:
			return
		if not isfinite(weight) or weight <= 0:
			raise ValueError("Progress work must be positive and finite.")
		if self._completed_steps + weight > self._total_steps:
			raise RuntimeError("Training progress exceeded the planned work count.")
		self._completed_steps += weight
		if self._stage is not None and self._completed_steps == self._total_steps:
			return
		milestone = int(
			self._completed_steps * 100 // (self._total_steps * self._milestone_percent)
		)
		if milestone < self._next_milestone:
			return
		self._next_milestone = milestone + 1
		self._print(description, stage=stage)
		if self._completed_steps == self._total_steps:
			self._finished = True

	def finish_stage(self, description: str) -> None:
		"""Print 100% once, after all planned work and caller checks succeed."""
		if self._stage is None or self._started is None:
			raise RuntimeError("No progress stage has been started.")
		if self._finished:
			return
		if self._completed_steps != self._total_steps:
			raise RuntimeError("Cannot finish a stage with incomplete work.")
		self._print(description, stage="")
		self._finished = True

	def _print(self, description: str, *, stage: str) -> None:
		"""Estimate this counter's total duration using only its elapsed work."""
		assert self._started is not None
		percentage = self._completed_steps * 100 / self._total_steps
		elapsed_seconds = perf_counter() - self._started
		estimated_seconds = elapsed_seconds * self._total_steps / self._completed_steps
		context = self._stage or stage
		if self._stage is not None and stage and stage.casefold() != context.casefold():
			context += f" ({stage})"
		print(
			f"{self._model} {self._mode} {context} {description}:\n"
			f"progress: {percentage:.1f}% "
			f"work: {self._completed_steps:,.1f}/{self._total_steps:,.1f} "
			f"time: {format_duration(elapsed_seconds)}/{format_duration(estimated_seconds)}",
			flush=True,
		)
