"""Run cross-validation for every YAML config in the project configs directory."""

from __future__ import annotations

import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def main() -> int:
    configs = sorted(
        path for path in CONFIG_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
    )
    if not configs:
        print(f"No YAML configs found in {CONFIG_DIR}")
        return 1

    failed_configs: list[str] = []
    for config in configs:
        print(f"Running models with {config.name}", flush=True)
        result = subprocess.run(
            ["uv", "run", "model", "ALL", "--config", str(config)],
            cwd=PROJECT_ROOT,
            check=False,
        )
        if result.returncode:
            failed_configs.append(config.name)

    print(f"Completed {len(configs) - len(failed_configs)}/{len(configs)} configs.")
    if failed_configs:
        print(f"Failed configs: {', '.join(failed_configs)}")
    return int(bool(failed_configs))


if __name__ == "__main__":
    raise SystemExit(main())
