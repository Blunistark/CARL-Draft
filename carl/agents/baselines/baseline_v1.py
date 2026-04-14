"""Simple baseline agent for benchmarking against DRL agents."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from carl.agents.base_agent import BaseAgent


class BaselineAgentV1(BaseAgent):
    """Rule-based / random baseline agent.

    Serves as a lower-bound performance reference when evaluating TD3 and
    other DRL agents.  By default uses a random policy; subclasses may
    override ``select_action`` with domain heuristics.

    Args:
        action_dim: Dimensionality of the action space.
        action_scale: Absolute bound applied symmetrically to sampled actions.
        seed: Optional random seed for reproducibility.
    """

    def __init__(
        self,
        action_dim: int,
        action_scale: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.action_dim = action_dim
        self.action_scale = action_scale
        self._rng = np.random.default_rng(seed)

    def select_action(self, state: Any) -> np.ndarray:
        """Return a uniformly random action within [-action_scale, action_scale].

        Args:
            state: Current environment observation (unused in random policy).

        Returns:
            Random action array of shape ``(action_dim,)``.
        """
        return self._rng.uniform(
            -self.action_scale, self.action_scale, size=(self.action_dim,)
        ).astype(np.float32)

    def update(self, batch: Any) -> dict[str, float]:
        """No-op update; baseline has no learnable parameters.

        Args:
            batch: Experience batch (ignored).

        Returns:
            Empty metrics dict.
        """
        return {}

    def save(self, path: str | Path) -> None:
        """No-op save; baseline carries no persistent state.

        Args:
            path: Target path (ignored).
        """

    def load(self, path: str | Path) -> None:
        """No-op load; baseline carries no persistent state.

        Args:
            path: Source path (ignored).
        """
