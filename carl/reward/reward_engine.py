"""Shaped reward computation for UAV path-following tasks."""
from __future__ import annotations

from typing import Any

import numpy as np


class RewardEngine:
    """Computes shaped rewards for fixed-wing UAV path-following.

    Combines three reward components:
      1. **Tracking** – negative cross-track and altitude error.
      2. **Smoothness** – penalty for large or jerky control deflections.
      3. **Survival** – bonus for staying within safe operational bounds.

    The relative contribution of each component is controlled by ``weights``.

    Args:
        weights: Dict with optional keys ``tracking``, ``smoothness``,
            ``survival``.  Missing keys default to ``1.0``.
        tracking_scale: Normalisation factor for tracking error (metres).
        smoothness_scale: Normalisation factor for action magnitude.
    """

    _DEFAULT_WEIGHTS: dict[str, float] = {
        "tracking": 1.0,
        "smoothness": 0.1,
        "survival": 0.5,
    }

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        tracking_scale: float = 10.0,
        smoothness_scale: float = 1.0,
    ) -> None:
        self.weights: dict[str, float] = {**self._DEFAULT_WEIGHTS, **(weights or {})}
        self.tracking_scale = tracking_scale
        self.smoothness_scale = smoothness_scale
        self._prev_action: np.ndarray | None = None

    def set_weights(self, weights: dict[str, float]) -> None:
        """Update reward component weights at runtime.

        Args:
            weights: Partial or full dict of new weight values.  Keys not
                present in ``weights`` retain their current values.
        """
        self.weights.update(weights)

    def compute(
        self,
        state: Any,
        action: np.ndarray,
        next_state: Any,
        info: dict[str, Any] | None = None,
    ) -> float:
        """Compute the total shaped reward for one environment transition.

        Args:
            state: Observation before the action was applied.
            action: Control action taken (flat array).
            next_state: Observation after the action was applied.
            info: Optional auxiliary info dict from the environment (may
                contain ``tracking_error``, ``altitude``, ``alive`` flags).

        Returns:
            Scalar total reward.
        """
        info = info or {}
        action_arr = np.asarray(action, dtype=np.float32)

        # --- Tracking component ---
        tracking_error: float = float(info.get("tracking_error", 0.0))
        r_tracking = -tracking_error / self.tracking_scale

        # --- Smoothness component ---
        if self._prev_action is not None:
            action_delta = float(np.mean(np.abs(action_arr - self._prev_action)))
        else:
            action_delta = float(np.mean(np.abs(action_arr)))
        r_smoothness = -action_delta / self.smoothness_scale
        self._prev_action = action_arr.copy()

        # --- Survival component ---
        alive: bool = bool(info.get("alive", True))
        r_survival = 1.0 if alive else -10.0

        total = (
            self.weights["tracking"] * r_tracking
            + self.weights["smoothness"] * r_smoothness
            + self.weights["survival"] * r_survival
        )
        return float(total)

    def reset(self) -> None:
        """Reset any stateful components (call at the start of each episode)."""
        self._prev_action = None
