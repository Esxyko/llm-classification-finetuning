"""Count actual, unpadded A/B input tokens in the saved training and validation partitions."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from llm_classification_finetuning.cli_config import add_config_argument
from llm_classification_finetuning.config import (
	MAX_VALIDATION_ROW_DEVIATION,
	AppConfig,
	LoRAModelConfig,
	ModelConfig,
)
from llm_classification_finetuning.errors import DataPreparationError
from llm_classification_finetuning.models import MODEL_REGISTRY
from llm_classification_finetuning.models.common import (
	ModelDataRepository,
	TokenizationCache,
)


class SplitTokenStatistics:
	"""Reuse training serialization and the saved holdout assignment for token counts."""

	def __init__(
		self,
		config: AppConfig,
		model: ModelConfig | LoRAModelConfig,
		*,
		batch_size: int,
		revision: str | None,
		local_files_only: bool,
	) -> None:
		self._config = config
		self._model = model
		self._batch_size = batch_size
		self._revision = revision
		self._local_files_only = local_files_only

	def run(self, output_dir: Path) -> None:
		"""Write per-ID counts, partition summaries, and reproducibility metadata."""
		from transformers import AutoConfig, AutoTokenizer

		repository = ModelDataRepository(
			self._config.data.processed_path,
			validation_fraction=self._config.validation.validation_fraction,
		)
		print(
			"Loading saved train/validation split and serializing canonical A/B inputs...",
			flush=True,
		)
		data = repository.load()
		base_config = AutoConfig.from_pretrained(
			self._model.model_name,
			revision=self._revision,
			local_files_only=self._local_files_only,
		)
		revision = getattr(base_config, "_commit_hash", None) or self._revision
		tokenizer = AutoTokenizer.from_pretrained(
			self._model.model_name,
			revision=revision,
			local_files_only=self._local_files_only,
		)
		print(f"Tokenizer: {self._model.model_name}; revision: {revision}", flush=True)
		cache = TokenizationCache(
			self._config.data.processed_path.parent / "token_cache", tokenizer
		)
		records = data.canonical_texts
		lengths = np.empty((len(records), 2), dtype=np.int64)
		next_progress = 5
		for start in range(0, len(records), self._batch_size):
			batch = records[start : start + self._batch_size]
			encoded = cache.encode(
				[text for row in batch for text in (row.response_a, row.response_b)],
				add_special_tokens=True,
				truncation=False,
			)
			stop = start + len(batch)
			lengths[start:stop] = np.diff(encoded.offsets).reshape(-1, 2)
			if stop * 100 >= len(records) * next_progress:
				print(f"Counted {stop:,}/{len(records):,} source IDs", flush=True)
				next_progress += 5
		rows = pd.DataFrame(
			{
				"id": [row.row_id for row in records],
				"tokens_a": lengths[:, 0],
				"tokens_b": lengths[:, 1],
				"tokens_total": lengths.sum(axis=1),
			}
		).merge(
			data.reference.loc[~data.reference["is_swapped"], ["id", "split", "label"]],
			on="id",
			validate="one_to_one",
		)
		splits = self._summarize(rows)
		metadata = {
			"processed_path": str(self._config.data.processed_path),
			"processed_sha256": data.fingerprint,
			"serialization": repository.serializer_version,
			"model_name": self._model.model_name,
			"tokenizer_revision": revision,
			"tokenizer_class": type(tokenizer).__name__,
			"transformers_version": version("transformers"),
			"tokenizers_version": version("tokenizers"),
			"add_special_tokens": True,
			"padding": False,
			"truncation": False,
			"configured_max_length": self._model.max_length,
			"ab_swap": self._model.ab_swap,
			"source_rows": len(rows),
			"tokens_total": int(lengths.sum()),
			"tokens_augmented": int(lengths.sum()) * 2,
			"split_balance": {
				"validation_fraction": self._config.validation.validation_fraction,
				"target_max_row_deviation": MAX_VALIDATION_ROW_DEVIATION,
				"measured_row_deviation": float(
					splits.loc[splits["split"].eq("validation"), "row_deviation_pct"]
					.abs()
					.max()
				)
				/ 100,
				"tokenizer_name": self._config.validation.tokenizer_name,
				"tokenizer_revision": self._config.validation.tokenizer_revision,
				"random_state": self._config.validation.random_state,
				"target_max_token_deviation": self._config.validation.max_token_deviation,
				"measured_max_token_deviation": float(
					splits.loc[splits["split"].eq("validation"), "token_deviation_pct"]
					.abs()
					.max()
				)
				/ 100,
			},
			"train_tokens_definition": (
				"Unpadded, untruncated training-partition tokens for one pass "
				"over the configured training orientations (aug=2x, inf=1x). "
				"Excludes epoch repetition, warmup, validation, and cache reuse."
			),
			"splits": splits.to_dict(orient="records"),
		}
		output_dir = output_dir.expanduser().resolve()
		output_dir.mkdir(parents=True, exist_ok=True)
		with TemporaryDirectory(prefix=".split-tokens-", dir=output_dir) as staging:
			staged = Path(staging)
			rows.to_csv(staged / "per_id.csv", index=False)
			splits.to_csv(staged / "splits.csv", index=False, float_format="%.8f")
			(staged / "summary.json").write_text(
				json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
				encoding="utf-8",
			)
			for path in sorted(staged.iterdir()):
				path.replace(output_dir / path.name)
		print(splits.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
		print(f"Total canonical tokens: {int(lengths.sum()):,}")
		print(f"Reports: {output_dir}")

	def _summarize(self, rows: pd.DataFrame) -> pd.DataFrame:
		total_tokens = int(rows["tokens_total"].sum())
		validation_fraction = self._config.validation.validation_fraction
		training_multiplier = 2 if self._model.ab_swap == "aug" else 1
		summaries = []
		for split, subset in rows.groupby("split", sort=True):
			fraction = (
				validation_fraction
				if split == "validation"
				else 1 - validation_fraction
			)
			multiplier = training_multiplier if split == "train" else 1
			branches = subset[["tokens_a", "tokens_b"]].to_numpy().reshape(-1)
			tokens = int(subset["tokens_total"].sum())
			counts = subset["label"].value_counts().reindex((0, 1, 2), fill_value=0)
			median, p95, p99 = np.quantile(branches, [0.5, 0.95, 0.99])
			summaries.append(
				{
					"split": split,
					"row_share_pct": len(subset) / len(rows) * 100,
					"row_deviation_pct": (len(subset) / (len(rows) * fraction) - 1)
					* 100,
					"source_rows": len(subset),
					"model_a_wins": int(counts.loc[0]),
					"model_b_wins": int(counts.loc[1]),
					"ties": int(counts.loc[2]),
					"tokens_a": int(subset["tokens_a"].sum()),
					"tokens_b": int(subset["tokens_b"].sum()),
					"tokens_total": tokens,
					"tokens_augmented": tokens * 2,
					"token_deviation_pct": (tokens / (total_tokens * fraction) - 1)
					* 100,
					"token_share_pct": tokens / total_tokens * 100
					if total_tokens
					else 0,
					"mean_pair_tokens": float(subset["tokens_total"].mean()),
					"median_branch_tokens": float(median),
					"p95_branch_tokens": float(p95),
					"p99_branch_tokens": float(p99),
					"max_branch_tokens": int(branches.max()),
					"branches_over_max_length": int(
						(branches > self._model.max_length).sum()
					),
					"effective_rows": len(subset) * multiplier,
					"effective_tokens": tokens * multiplier,
				}
			)
		return pd.DataFrame(summaries)


def main() -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	add_config_argument(parser)
	parser.add_argument("--model", choices=MODEL_REGISTRY, default="qwen3-4b-lora")
	parser.add_argument("--batch-size", type=int, default=256)
	parser.add_argument("--revision", help="Optional Hub tokenizer/base revision.")
	parser.add_argument("--local-files-only", action="store_true")
	parser.add_argument("--output-dir", type=Path)
	args = parser.parse_args()
	if args.batch_size < 1:
		parser.error("--batch-size must be positive.")
	try:
		config = AppConfig.load(args.config)
		model = MODEL_REGISTRY[args.model].profile.resolve_config(config)
		revision = args.revision
		if revision is None and model.model_name == config.validation.tokenizer_name:
			revision = config.validation.tokenizer_revision
		SplitTokenStatistics(
			config,
			model,
			batch_size=args.batch_size,
			revision=revision,
			local_files_only=args.local_files_only,
		).run(
			args.output_dir or config.data.processed_path.parent / "split_token_stats"
		)
	except (DataPreparationError, OSError, ValueError) as error:
		print(f"Could not count split tokens: {error}")
		return 1
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
