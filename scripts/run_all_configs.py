"""Cross-validate and build a separate head for every project YAML config."""

from __future__ import annotations

import re
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

    tags = [config.stem for config in configs]
    if any(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", tag) is None for tag in tags):
        print("Config filenames must have stems valid as checkpoint tags.")
        return 1
    if len(tags) != len({tag.casefold() for tag in tags}):
        print("Config filenames must have distinct stems for checkpoint names.")
        return 1

    failed_configs: list[str] = []
    for config in configs:
        print(f"Running models with {config.name}", flush=True)
        failed = False
        for mode in ("cross-validation", "build"):
            command = ["uv", "run", "model", "ALL", "--config", str(config)]
            if mode == "build":
                command.extend(("--build", "--checkpoint-tag", config.stem))
            print(f"Running {mode} for {config.name}", flush=True)
            result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
            if result.returncode:
                failed = True
                print(f"Failed {mode} for {config.name}", flush=True)
        if failed:
            failed_configs.append(config.name)

    print(f"Completed {len(configs) - len(failed_configs)}/{len(configs)} configs.")
    if failed_configs:
        print(f"Failed configs: {', '.join(failed_configs)}")
    return int(bool(failed_configs))


if __name__ == "__main__":
    raise SystemExit(main())
