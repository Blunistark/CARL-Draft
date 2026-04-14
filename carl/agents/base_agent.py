"""Base reinforcement learning agent interface for CARL."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseAgent(ABC):
    """Abstract base class defining the interface for all CARL RL agents.

    All concrete agent implementations (TD3, baseline, etc.) must subclass
    this and implement the four core lifecycle methods.
    """

    @abstractmethod
    def select_action(self, state: Any) -> Any:
        """Select an action given the current environment state.

        Args:
            state: The current observation from the environment.

        Returns:
            The action to be taken.
        """

    @abstractmethod
    def update(self, batch: Any) -> dict[str, float]:
        """Update agent parameters from a sampled experience batch.

        Args:
            batch: A batch of transitions (state, action, reward, next_state, done).

        Returns:
            A dict of scalar training metrics (e.g. actor_loss, critic_loss).
        """

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist agent weights and state to disk.

        Args:
            path: Directory or file path to write checkpoint to.
        """

    @abstractmethod
    def load(self, path: str | Path) -> None:
        """Restore agent weights and state from disk.

        Args:
            path: Directory or file path to read checkpoint from.
        """
