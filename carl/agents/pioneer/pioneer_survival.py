"""Pioneer agent augmented with survival (crash-avoidance) objectives."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from carl.agents.pioneer.pioneer_v1 import PioneerAgentV1


class PioneerSurvivalAgent(PioneerAgentV1):
    """TD3 Pioneer agent extended with explicit survival shaping.

    Builds on :class:`PioneerAgentV1` and adds a survival bonus / penalty
    term that discourages flight conditions likely to lead to crashes (e.g.
    extreme pitch, stall speed, proximity to ground).

    Args:
        state_dim: Dimension of the observation space.
        action_dim: Dimension of the action space.
        survival_weight: Scalar multiplier applied to the survival reward
            component when blending with the path-following reward.
        min_altitude: Altitude threshold (metres) below which a survival
            penalty is incurred.
        **kwargs: Additional keyword arguments forwarded to
            :class:`PioneerAgentV1`.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        survival_weight: float = 0.5,
        min_altitude: float = 10.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(state_dim, action_dim, **kwargs)
        self.survival_weight = survival_weight
        self.min_altitude = min_altitude

    def _survival_penalty(self, state: Any) -> float:
        """Compute survival penalty from the current state.

        Args:
            state: Current environment observation.

        Returns:
            Non-positive penalty scalar.
        """
        raise NotImplementedError

    def select_action(self, state: Any) -> np.ndarray:
        """Select action with optional survival-aware exploration noise.

        Args:
            state: Current environment observation.

        Returns:
            Action array of shape ``(action_dim,)``.
        """
        raise NotImplementedError

    def update(self, batch: Any) -> dict[str, float]:
        """TD3 update with survival-augmented reward signal.

        Args:
            batch: Tuple of (states, actions, rewards, next_states, dones).

        Returns:
            Dict containing ``actor_loss``, ``critic_loss``, and
            ``survival_penalty`` scalars.
        """
        raise NotImplementedError
