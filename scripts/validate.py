"""Validate each model on the saved holdout using its selected YAML configs."""

from __future__ import annotations

import subprocess
from pathlib import Path

CONFIG_DIR = "configs"
MODEL_CONFIGS: dict[str, list[str]] = {
	"SR2-Qwen3-4B-lora": ["1.yaml", "2.yaml", "3.yaml", "4.yaml"],
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
	total_tasks = sum(len(configs) for configs in MODEL_CONFIGS.values())
	if not total_tasks:
		print("No configs selected in MODEL_CONFIGS.")
		return 1

	config_dir = PROJECT_ROOT / CONFIG_DIR
	failed_tasks: list[str] = []
	for model, configs in MODEL_CONFIGS.items():
		for filename in configs:
			config = config_dir / filename
			task = f"{filename}: {model}"
			if not config.is_file():
				failed_tasks.append(task)
				print(f"Config file not found for {task}: {config}", flush=True)
				continue
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

	print(f"Completed {total_tasks - len(failed_tasks)}/{total_tasks} tasks.")
	if failed_tasks:
		print(f"Failed tasks: {', '.join(failed_tasks)}")
	return int(bool(failed_tasks))


if __name__ == "__main__":
	raise SystemExit(main())
