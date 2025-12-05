"""Utilities for loading and validating project configuration."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load YAML config and return it as a mutable dictionary."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    data["config_path"] = str(config_path)
    default_root = (
        config_path.parent.parent
        if config_path.parent.name == "configs"
        else config_path.parent
    )
    data.setdefault("project_root", str(default_root))
    return data


def apply_overrides(
    config: MutableMapping[str, Any], overrides: Mapping[str, Any] | None = None
) -> Dict[str, Any]:
    """Deep-merge overrides into the config tree."""

    if not overrides:
        return config

    def _merge(target: MutableMapping[str, Any], patch: Mapping[str, Any]) -> None:
        for key, value in patch.items():
            if (
                isinstance(value, Mapping)
                and key in target
                and isinstance(target[key], MutableMapping)
            ):
                _merge(target[key], value)
            else:
                target[key] = value

    _merge(config, overrides)
    return config


def to_path(value: str | Path, base_dir: str | Path | None = None) -> Path:
    """Convert a config value to an absolute path."""
    candidate = Path(value)
    if candidate.is_absolute() or base_dir is None:
        return candidate
    return (Path(base_dir) / candidate).resolve()


def ensure_parent(path: Path) -> Path:
    """Create parent directories for the provided path if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
