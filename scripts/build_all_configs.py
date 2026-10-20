"""Build selected models on all training data for every project YAML config."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

MODELS = ("qwen3-4b-lora",)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def main() -> int:
	if not MODELS:
		print("No models selected in MODELS.")
		return 1

	configs = sorted(
		path
		for path in CONFIG_DIR.glob("*")
		if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
	)
	if not configs:
		print(f"No YAML configs found in {CONFIG_DIR}")
		return 1

	tags = [config.stem for config in configs]
	if any(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", tag) is None for tag in tags):
		print("Config filenames must have stems valid as checkpoint tags.")
		return 1
	if len(tags) != len({tag.casefold() for tag in tags}):
		print("Config filenames must have distinct stems for checkpoint names.")
		return 1

	failed_tasks: list[str] = []
	for config in configs:
		for model in MODELS:
			task = f"{config.name}: {model}"
			command = [
				"uv",
				"run",
				"model",
				model,
				"--config",
				str(config),
				"--build",
				"--checkpoint-tag",
				config.stem,
			]
			print(f"Running full-data build for {task}", flush=True)
			try:
				result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
			except OSError as error:
				failed_tasks.append(task)
				print(f"Failed full-data build for {task}: {error}", flush=True)
				continue
			if result.returncode:
				failed_tasks.append(task)
				print(f"Failed full-data build for {task}", flush=True)

	total_tasks = len(configs) * len(MODELS)
	print(f"Completed {total_tasks - len(failed_tasks)}/{total_tasks} tasks.")
	if failed_tasks:
		print(f"Failed tasks: {', '.join(failed_tasks)}")
	return int(bool(failed_tasks))


if __name__ == "__main__":
	raise SystemExit(main())
