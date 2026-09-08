from __future__ import annotations

import platform
import random
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from .artifacts import atomic_json_write, stable_hash
from .config import ExperimentConfig


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def initialize_run(config: ExperimentConfig, command: str) -> None:
    """Seed available RNGs and persist the exact runtime/configuration record."""
    random.seed(config.experiment.seed)
    runtime: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": _package_version("torch"),
        "transformers": _package_version("transformers"),
        "pyyaml": _package_version("PyYAML"),
    }
    try:
        import torch

        torch.manual_seed(config.experiment.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.experiment.seed)
            runtime["cuda"] = {
                "version": torch.version.cuda,
                "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
            }
        if config.experiment.deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
    payload = {
        "schema_version": 1,
        "experiment": config.experiment.name,
        "command": command,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config.to_dict(),
        "config_digest": stable_hash(config.to_dict()),
        "runtime": runtime,
    }
    atomic_json_write(config.output_dir / "run_manifest.json", payload)
