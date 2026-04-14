"""Configuration loading utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a configuration file into a dictionary.

    Supports YAML (``.yaml`` / ``.yml``) and JSON (``.json``) formats.
    Returns a plain ``dict`` so that callers can freely add or override keys.

    Args:
        path: Path to the configuration file.

    Returns:
        Configuration as a mutable dictionary.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file extension is not recognised.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-untyped]
            with path.open() as f:
                return yaml.safe_load(f) or {}
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load YAML configs: pip install pyyaml"
            ) from exc
    elif suffix == ".json":
        with path.open() as f:
            return json.load(f)
    else:
        raise ValueError(
            f"Unsupported config format '{suffix}'. Use .yaml, .yml, or .json."
        )
