"""Path helpers for reproducible file resolution across the CARL project."""
from __future__ import annotations

from pathlib import Path


def get_project_root() -> Path:
    """Return the absolute path to the repository root.

    The root is resolved as the directory three levels above this file
    (``carl/utils/paths.py`` → ``carl/utils/`` → ``carl/`` → repo root).

    Returns:
        :class:`~pathlib.Path` pointing to the repository root directory.
    """
    return Path(__file__).resolve().parents[2]


def get_run_dir(run_name: str) -> Path:
    """Return the output directory for a named training run.

    Args:
        run_name: Human-readable identifier for the run (e.g.
            ``"td3_pioneer_stage3"``).

    Returns:
        :class:`~pathlib.Path` at ``<project_root>/outputs/runs/<run_name>``.
    """
    return get_project_root() / "outputs" / "runs" / run_name


def get_checkpoint_dir(run_name: str) -> Path:
    """Return the checkpoint sub-directory for a named training run.

    Args:
        run_name: Identifier for the run.

    Returns:
        :class:`~pathlib.Path` at
        ``<project_root>/outputs/runs/<run_name>/checkpoints``.
    """
    return get_run_dir(run_name) / "checkpoints"


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and all parents) if it does not already exist.

    Args:
        path: Directory path to create.

    Returns:
        The resolved :class:`~pathlib.Path` (always absolute).
    """
    resolved = Path(path).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
