"""Validate selected models on the saved holdout for every project YAML config."""

from __future__ import annotations

import subprocess
from pathlib import Path

MODELS = ("qwen3-4b-lora-EXPT2", "SR2-Qwen3-4B-lora")

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

	failed_tasks: list[str] = []
	for config in configs:
		for model in MODELS:
			task = f"{config.name}: {model}"
			command = ["uv", "run", "model", model, "--config", str(config)]
			print(f"Running holdout validation for {task}", flush=True)
			try:
				result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
			except OSError as error:
				failed_tasks.append(task)
				print(f"Failed holdout validation for {task}: {error}", flush=True)
				continue
			if result.returncode:
				failed_tasks.append(task)
				print(f"Failed holdout validation for {task}", flush=True)

	total_tasks = len(configs) * len(MODELS)
	print(f"Completed {total_tasks - len(failed_tasks)}/{total_tasks} tasks.")
	if failed_tasks:
		print(f"Failed tasks: {', '.join(failed_tasks)}")
	return int(bool(failed_tasks))


if __name__ == "__main__":
	raise SystemExit(main())
