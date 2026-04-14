"""Staged curriculum learning manager with difficulty progression."""
from __future__ import annotations

from typing import Any


class CurriculumManager:
    """Manages a sequence of training stages with progressive difficulty.

    Each stage is defined by a configuration dict (e.g. wind speed range,
    waypoint distance, number of obstacles).  The manager tracks episode
    outcomes and advances to the next stage when a promotion criterion is met.

    Args:
        stages: Ordered list of stage configuration dicts.
        promotion_threshold: Fraction of recent episodes that must succeed
            before advancing to the next stage.
        window_size: Number of recent episodes considered for the promotion
            criterion.
    """

    def __init__(
        self,
        stages: list[dict[str, Any]],
        promotion_threshold: float = 0.7,
        window_size: int = 100,
    ) -> None:
        self.stages = stages
        self.promotion_threshold = promotion_threshold
        self.window_size = window_size
        self._current_stage: int = 0
        self._history: list[bool] = []

    def get_stage(self) -> dict[str, Any]:
        """Return the configuration dict for the current stage.

        Returns:
            Stage configuration mapping.
        """
        return self.stages[self._current_stage]

    def advance_stage(self) -> bool:
        """Attempt to move to the next stage.

        Returns:
            ``True`` if successfully advanced, ``False`` if already at the
            final stage or promotion criterion is not met.
        """
        if self._current_stage >= len(self.stages) - 1:
            return False
        self._current_stage += 1
        return True

    def record_episode(self, success: bool) -> None:
        """Record the outcome of a completed episode.

        Automatically checks and applies the promotion criterion after
        appending the result.

        Args:
            success: Whether the episode was considered successful.
        """
        self._history.append(success)
        recent = self._history[-self.window_size :]
        if len(recent) >= self.window_size:
            rate = sum(recent) / len(recent)
            if rate >= self.promotion_threshold:
                self.advance_stage()

    def reset(self) -> None:
        """Reset manager to stage 0 and clear episode history."""
        self._current_stage = 0
        self._history = []

    def get_config(self) -> dict[str, Any]:
        """Return a serialisable snapshot of the manager state.

        Returns:
            Dict with ``current_stage``, ``num_stages``, ``history_len``,
            and ``promotion_threshold`` keys.
        """
        return {
            "current_stage": self._current_stage,
            "num_stages": len(self.stages),
            "history_len": len(self._history),
            "promotion_threshold": self.promotion_threshold,
        }
