"""Run selected YAML configs for holdout validation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import traceback
from collections.abc import Sequence
from contextlib import redirect_stderr, redirect_stdout
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

from dotenv import load_dotenv

CONFIG_DIR = "configs"
MODEL_CONFIGS: dict[str, list[str]] = {
	"SR2-Qwen3-4B": ["lmsys33k.yaml"],
	# "SR2-Qwen3-4B-lora": ["lmsys33k.yaml"],
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _TeeStream:
	"""Write terminal output to the run log immediately as well."""

	def __init__(self, console: TextIO, log_file: TextIO) -> None:
		self._console = console
		self._log_file = log_file

	def write(self, text: str) -> int:
		self._log_file.write(text)
		self._log_file.flush()
		return self._console.write(text)

	def flush(self) -> None:
		self._log_file.flush()
		self._console.flush()


def _validate(data_source_root: Path | None = None) -> int:
	total_tasks = sum(len(configs) for configs in MODEL_CONFIGS.values())
	if not total_tasks:
		print("No configs selected in MODEL_CONFIGS.")
		return 1

	from llm_classification_finetuning.config import AppConfig
	from llm_classification_finetuning.errors import DataPreparationError
	from llm_classification_finetuning.pipeline import EnsureProcessedDataPipeline

	config_dir = PROJECT_ROOT / CONFIG_DIR
	failed_tasks: list[str] = []
	prepared_paths: set[Path] = set()
	for model, configs in MODEL_CONFIGS.items():
		for filename in configs:
			config = config_dir / filename
			task = f"{filename}: {model}"
			if not config.is_file():
				failed_tasks.append(task)
				print(f"Config file not found for {task}: {config}", flush=True)
				continue
			try:
				app_config = AppConfig.load(config)
				processed_path = app_config.data.processed_path
				if processed_path not in prepared_paths:
					print(
						f"Preparing processed data for {task} ({config}): {processed_path}",
						flush=True,
					)
					copied = EnsureProcessedDataPipeline(
						app_config,
						project_root=PROJECT_ROOT,
						source_root=data_source_root,
					).run(
						on_copy=lambda source, target: print(
							f"Copying processed data: {source} -> {target}", flush=True
						)
					)
					prepared_paths.add(processed_path)
					print(
						f"Processed data ready: {processed_path}; copied {len(copied)} files.",
						flush=True,
					)
			except (DataPreparationError, OSError) as error:
				failed_tasks.append(task)
				print(
					f"Failed data preparation for {task} ({config}): {error}",
					flush=True,
				)
				continue
			command = ["uv", "run", "model", model, "--config", str(config)]
			print(f"Running holdout validation for {task}", flush=True)
			try:
				with subprocess.Popen(
					command,
					cwd=PROJECT_ROOT,
					stdout=subprocess.PIPE,
					stderr=subprocess.STDOUT,
					text=True,
					encoding="utf-8",
					errors="replace",
					bufsize=1,
					env=os.environ
					| {"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
				) as process:
					assert process.stdout is not None
					for line in process.stdout:
						print(line, end="", flush=True)
					returncode = process.wait()
			except OSError as error:
				prepared_paths.discard(processed_path)
				failed_tasks.append(task)
				print(f"Failed holdout validation for {task}: {error}", flush=True)
				continue
			if returncode:
				# A failing loader can archive data; later tasks must check it again.
				prepared_paths.discard(processed_path)
				failed_tasks.append(task)
				print(
					f"Failed holdout validation for {task} (exit code {returncode})",
					flush=True,
				)

	print(f"Completed {total_tasks - len(failed_tasks)}/{total_tasks} tasks.")
	if failed_tasks:
		print(f"Failed tasks: {', '.join(failed_tasks)}")
	return int(bool(failed_tasks))


def main(arguments: Sequence[str] | None = None) -> int:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument(
		"--data-source-root",
		type=Path,
		help="Optional project mirror used to fill missing processed-data files.",
	)
	parsed_arguments = parser.parse_args(arguments)
	os.chdir(PROJECT_ROOT)
	load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
	from llm_classification_finetuning.config import resolve_result_dir

	log_dir = resolve_result_dir(PROJECT_ROOT) / "logs"
	log_dir.mkdir(parents=True, exist_ok=True)
	generated_at = datetime.now(UTC).astimezone()
	log_path = log_dir / f"validate-{generated_at:%Y%m%d-%H%M%S-%f}.log"
	with (
		log_path.open("x", encoding="utf-8") as log_file,
		redirect_stdout(_TeeStream(sys.stdout, log_file)),
		redirect_stderr(_TeeStream(sys.stderr, log_file)),
	):
		print(f"Saving validation log to {log_path}", flush=True)
		try:
			return _validate(parsed_arguments.data_source_root)
		except BaseException:
			traceback.print_exc(file=log_file)
			raise


if __name__ == "__main__":
	raise SystemExit(main())
