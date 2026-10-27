"""Single-branch recompute: cache branch A, backpropagate B/head, then recompute A."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import Tensor

from .data import PairwiseBatch
from .model import LoRAQwenClassifier


@dataclass(frozen=True, slots=True)
class RandomState:
	"""CPU and CUDA generator states captured before a stochastic forward."""

	cpu: Tensor
	cuda: Tensor | None
	device: torch.device

	@classmethod
	def capture(cls, device: torch.device) -> RandomState:
		"""Snapshot the CPU generator and, for CUDA devices, that device's generator."""
		cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
		return cls(torch.get_rng_state(), cuda, device)

	@contextmanager
	def replay(self) -> Iterator[None]:
		"""Replay this state, then restore the caller's training RNG even on failure."""
		devices = [self.device] if self.cuda is not None else []
		with torch.random.fork_rng(devices=devices, device_type="cuda"):
			torch.set_rng_state(self.cpu)
			if self.cuda is not None:
				torch.cuda.set_rng_state(self.cuda, self.device)
			yield


class SingleBranchRecompute:
	"""Hold only branch A's features while B and the head build the autograd graph."""

	def __init__(self, model: LoRAQwenClassifier, batch: PairwiseBatch) -> None:
		self._model = model
		self._batch = batch
		self._state: RandomState | None = None
		self._features: Tensor | None = None

	def encode(self) -> tuple[Tensor, Tensor]:
		"""Return A as a differentiable leaf and B with its graph.

		The checkpointing policy is chosen here and kept through both backward passes.
		"""
		model, batch = self._model, self._batch
		model.configure_joint_checkpointing(batch)
		self._state = RandomState.capture(model.device)
		with torch.no_grad():
			features = model.run_branch(batch.branch_a)
		self._features = features.detach().requires_grad_(True)
		return self._features, model.run_branch(batch.branch_b)

	def backward_a(self) -> None:
		"""Replay A's RNG, recompute it with the same policy, and apply cached grads."""
		features, state = self._features, self._state
		if features is None or state is None:
			raise RuntimeError("Branch A was not encoded before its recompute.")
		gradient = features.grad
		if gradient is None:
			raise RuntimeError("Backpropagate the head and branch B before branch A.")
		try:
			with state.replay():
				recomputed = self._model.run_branch(self._batch.branch_a)
			if recomputed.shape != gradient.shape or recomputed.dtype != gradient.dtype:
				raise RuntimeError("Recomputed branch A features changed shape or dtype.")
			torch.autograd.backward(recomputed, gradient)
		finally:
			self._features = None
			self._state = None

