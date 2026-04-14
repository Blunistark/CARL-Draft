"""TD3-based Pioneer agent for fixed-wing UAV path following."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from carl.agents.base_agent import BaseAgent


class PioneerAgentV1(BaseAgent):
    """Twin-Delayed Deep Deterministic Policy Gradient (TD3) agent.

    Implements the TD3 algorithm [Fujimoto et al., 2018] for continuous
    control of a fixed-wing UAV along a reference trajectory.

    Args:
        state_dim: Dimension of the observation space.
        action_dim: Dimension of the action space.
        hidden_dim: Width of hidden MLP layers in actor / critic networks.
        actor_lr: Learning rate for the actor optimiser.
        critic_lr: Learning rate for the critic optimiser.
        gamma: Discount factor.
        tau: Soft update coefficient for target networks.
        policy_noise: Std of Gaussian noise added to target policy actions.
        noise_clip: Range to clip target policy noise.
        policy_delay: Number of critic updates per actor update.
        device: Torch device string (e.g. ``"cpu"`` or ``"cuda"``).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        device: str = "cpu",
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.device = device
        self._total_updates = 0

        # Networks and optimisers are initialised lazily on first use
        self._actor = None
        self._critic = None
        self._actor_target = None
        self._critic_target = None
        self._actor_optim = None
        self._critic_optim = None

    def _build_networks(self) -> None:
        """Instantiate actor, critic and their target networks."""
        raise NotImplementedError("Network construction not yet implemented.")

    def select_action(self, state: Any) -> np.ndarray:
        """Select a deterministic action from the actor network.

        Args:
            state: Current environment observation.

        Returns:
            Action array of shape ``(action_dim,)``.
        """
        raise NotImplementedError

    def update(self, batch: Any) -> dict[str, float]:
        """Perform one TD3 update step on the provided experience batch.

        Args:
            batch: Tuple of (states, actions, rewards, next_states, dones).

        Returns:
            Dict containing ``actor_loss`` and ``critic_loss`` scalars.
        """
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        """Save actor, critic and target network weights to ``path``.

        Args:
            path: Destination directory for checkpoint files.
        """
        raise NotImplementedError

    def load(self, path: str | Path) -> None:
        """Load actor, critic and target network weights from ``path``.

        Args:
            path: Source directory containing checkpoint files.
        """
        raise NotImplementedError
