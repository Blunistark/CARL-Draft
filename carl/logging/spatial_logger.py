"""Spatial trajectory logger for recording UAV episode data."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class SpatialLogger:
    """Records spatial trajectory data (3-D positions, velocities, actions) during episodes.

    Data is accumulated in memory and can be flushed to disk as JSON or
    NumPy ``.npz`` files.

    Args:
        episode_id: Optional identifier attached to all log entries.
        buffer_size: Maximum number of steps buffered before an auto-flush
            (0 = unlimited).
    """

    def __init__(
        self,
        episode_id: str | int | None = None,
        buffer_size: int = 0,
    ) -> None:
        self.episode_id = episode_id
        self.buffer_size = buffer_size
        self._steps: list[dict[str, Any]] = []

    def log_step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        info: dict[str, Any] | None = None,
    ) -> None:
        """Append a single transition to the trajectory buffer.

        Args:
            state: Current environment observation / state vector.
            action: Action taken at this step.
            reward: Scalar reward received.
            info: Optional auxiliary info dict from the environment.
        """
        entry: dict[str, Any] = {
            "step": len(self._steps),
            "state": np.asarray(state).tolist(),
            "action": np.asarray(action).tolist(),
            "reward": float(reward),
        }
        if info:
            entry["info"] = info
        self._steps.append(entry)

        if self.buffer_size > 0 and len(self._steps) >= self.buffer_size:
            # Caller is responsible for calling save; just warn here.
            pass

    def clear(self) -> None:
        """Clear the in-memory trajectory buffer."""
        self._steps = []

    def save(self, path: str | Path) -> None:
        """Persist the trajectory buffer to disk.

        Saves as JSON if ``path`` ends with ``.json``, otherwise as NumPy
        ``.npz``.

        Args:
            path: Destination file path (parent directory must exist).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix == ".json":
            payload = {
                "episode_id": self.episode_id,
                "steps": self._steps,
            }
            path.write_text(json.dumps(payload, indent=2))
        else:
            states = np.array([s["state"] for s in self._steps], dtype=np.float32)
            actions = np.array([s["action"] for s in self._steps], dtype=np.float32)
            rewards = np.array([s["reward"] for s in self._steps], dtype=np.float32)
            np.savez_compressed(path, states=states, actions=actions, rewards=rewards)

    @classmethod
    def load(cls, path: str | Path) -> "SpatialLogger":
        """Load a previously saved trajectory from disk.

        Args:
            path: Source file path (``.json`` or ``.npz``).

        Returns:
            A new :class:`SpatialLogger` instance populated with the saved data.
        """
        path = Path(path)
        logger = cls()

        if path.suffix == ".json":
            payload = json.loads(path.read_text())
            logger.episode_id = payload.get("episode_id")
            logger._steps = payload.get("steps", [])
        else:
            data = np.load(path)
            for i, (s, a, r) in enumerate(
                zip(data["states"], data["actions"], data["rewards"])
            ):
                logger._steps.append(
                    {"step": i, "state": s.tolist(), "action": a.tolist(), "reward": float(r)}
                )
        return logger

    def __len__(self) -> int:
        return len(self._steps)
