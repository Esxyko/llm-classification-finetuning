"""Load and serialize processed pairwise training records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import PreTrainedTokenizerBase

from ...errors import ModelExecutionError
from .inputs import TrainingInputLoader


@dataclass(frozen=True, slots=True)
class PairwiseText:
	"""One canonical source record serialized into two model inputs."""

	row_id: int
	response_a: str
	response_b: str


@dataclass(frozen=True, slots=True)
class ModelData:
	"""Validated augmented references and canonical model inputs."""

	reference: pd.DataFrame
	canonical_texts: tuple[PairwiseText, ...]
	fingerprint: str


class ConversationSerializer:
	"""Render aligned user/assistant turns with the supplied model's chat template."""

	FORMAT_VERSION = "sr2-qwen3-chat-template-v1"

	def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
		self._tokenizer = tokenizer
		try:
			self._chat_template = tokenizer.get_chat_template()
			if not self._chat_template:
				raise ValueError("the tokenizer has no chat template")
		except Exception as error:
			raise ModelExecutionError(
				f"Could not prepare the conversation chat template: {error}"
			) from error

	def serialize_pair(
		self,
		row_id: int,
		prompt_value: str,
		response_a_value: str,
		response_b_value: str,
	) -> PairwiseText:
		"""Build the A and B branch text for one canonical row."""
		prompts = self._load_array(prompt_value, row_id, "prompt")
		response_a = self._load_array(response_a_value, row_id, "response_a")
		response_b = self._load_array(response_b_value, row_id, "response_b")

		if not prompts:
			raise ModelExecutionError(
				f"Row {row_id} contains an empty prompt conversation."
			)
		if len(prompts) != len(response_a) or len(prompts) != len(response_b):
			raise ModelExecutionError(
				f"Row {row_id} has misaligned prompt and response turn counts."
			)

		for turn, prompt in enumerate(prompts):
			if not isinstance(prompt, str):
				raise ModelExecutionError(
					f"Row {row_id} prompt turn {turn} must be a string."
				)

		text_a = self._serialize_branch(row_id, prompts, response_a, "response_a")
		text_b = self._serialize_branch(row_id, prompts, response_b, "response_b")
		return PairwiseText(row_id=row_id, response_a=text_a, response_b=text_b)

	@staticmethod
	def _load_array(value: str, row_id: int, column: str) -> list[Any]:
		if not isinstance(value, str):
			raise ModelExecutionError(f"Row {row_id} {column} must be JSON text.")
		try:
			parsed = json.loads(value)
		except (TypeError, json.JSONDecodeError) as error:
			raise ModelExecutionError(
				f"Row {row_id} {column} is not a valid JSON array: {error}"
			) from error
		if not isinstance(parsed, list):
			raise ModelExecutionError(f"Row {row_id} {column} must be a JSON array.")
		return parsed

	def _serialize_branch(
		self,
		row_id: int,
		prompts: list[Any],
		responses: list[Any],
		column: str,
	) -> str:
		messages: list[dict[str, str]] = []
		for turn, (prompt, response) in enumerate(zip(prompts, responses, strict=True)):
			if response is not None and not isinstance(response, str):
				raise ModelExecutionError(
					f"Row {row_id} {column} turn {turn} must be a string or null."
				)
			messages.extend(
				(
					{"role": "user", "content": prompt},
					{"role": "assistant", "content": response or ""},
				)
			)
		try:
			return self._tokenizer.apply_chat_template(
				messages,
				chat_template=self._chat_template,
				tokenize=False,
				add_generation_prompt=False,
				continue_final_message=False,
			)
		except Exception as error:
			raise ModelExecutionError(
				f"Could not apply the chat template to row {row_id} {column}: {error}"
			) from error


class ModelDataRepository:
	"""Read and validate the processed holdout artifact for model training."""

	REQUIRED_COLUMNS = TrainingInputLoader.REQUIRED_COLUMNS
	SWAPPED_LABELS = TrainingInputLoader.SWAPPED_LABELS

	def __init__(
		self,
		processed_path: Path,
		*,
		serializer: ConversationSerializer,
		validation_fraction: float | None = None,
	) -> None:
		self._loader = TrainingInputLoader(
			processed_path, validation_fraction=validation_fraction
		)
		self._serializer = serializer

	@property
	def serializer_version(self) -> str:
		"""Return the cache-relevant conversation format identifier."""
		return self._serializer.FORMAT_VERSION

	def load(self) -> ModelData:
		"""Return validated references, canonical texts, and a file fingerprint."""
		inputs = self._loader.load()
		data = inputs.data
		canonical = data.loc[~data["is_swapped"]]
		canonical_texts = tuple(
			self._serializer.serialize_pair(
				row_id=int(row.id),
				prompt_value=row.prompt,
				response_a_value=row.response_a,
				response_b_value=row.response_b,
			)
			for row in canonical.itertuples(index=False)
		)
		fingerprint = inputs.fingerprint
		return ModelData(
			reference=data.loc[:, ["id", "label", "split", "is_swapped"]]
			.copy()
			.reset_index(drop=True),
			canonical_texts=canonical_texts,
			fingerprint=fingerprint,
		)
