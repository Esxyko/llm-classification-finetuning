"""Generate the audited LMSYS33K build-only synthetic tie supplement."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from llm_classification_finetuning.cli_config import add_config_argument
from llm_classification_finetuning.config import AppConfig
from llm_classification_finetuning.errors import DataPreparationError
from llm_classification_finetuning.preprocess import SyntheticTieGenerator


def main(arguments: Sequence[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	add_config_argument(parser)
	arguments = parser.parse_args(arguments)
	try:
		output = SyntheticTieGenerator(AppConfig.load(arguments.config)).run()
		print(f"Generation summary: {output / 'summary.json'}")
		return 0
	except (DataPreparationError, OSError) as error:
		print(f"Synthetic tie generation failed: {error}")
		return 1


if __name__ == "__main__":
	raise SystemExit(main())
