"""Fixed-wing UAV path-following Gym environment."""
from __future__ import annotations

from typing import Any

try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_AVAILABLE = True
    _GymEnv = gym.Env
except ImportError:
    try:
        import gym  # type: ignore[no-redef]
        from gym import spaces  # type: ignore[assignment]
        _GYM_AVAILABLE = True
        _GymEnv = gym.Env  # type: ignore[assignment]
    except ImportError:
        _GYM_AVAILABLE = False
        _GymEnv = object  # type: ignore[assignment,misc]

import numpy as np


class UAVEnv(_GymEnv):  # type: ignore[valid-type]
    """Fixed-wing UAV path-following environment.

    The agent controls elevator, aileron, rudder, and throttle deflections to
    track a reference waypoint trajectory in 3-D space.  Observations encode
    aircraft state (position, velocity, attitude) relative to the current
    waypoint.

    Args:
        dt: Simulation timestep in seconds.
        max_steps: Maximum episode length in steps.
        observation_dim: Dimension of the flat observation vector.
        action_dim: Dimension of the continuous action vector.
        render_mode: Render mode (``"human"`` or ``None``).
    """

    metadata: dict[str, Any] = {"render_modes": ["human"]}

    def __init__(
        self,
        dt: float = 0.05,
        max_steps: int = 1000,
        observation_dim: int = 24,
        action_dim: int = 4,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.dt = dt
        self.max_steps = max_steps
        self.render_mode = render_mode
        self._step_count: int = 0

        if _GYM_AVAILABLE:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(observation_dim,),
                dtype=np.float32,
            )
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(action_dim,),
                dtype=np.float32,
            )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to an initial state.

        Args:
            seed: Optional seed for the environment's RNG.
            options: Optional configuration overrides for the episode.

        Returns:
            Tuple of ``(observation, info)`` where ``observation`` is the
            initial flat state vector and ``info`` is an auxiliary dict.
        """
        if _GYM_AVAILABLE and hasattr(super(), "reset"):
            super().reset(seed=seed)

        self._step_count = 0
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        info: dict[str, Any] = {}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the simulation by one timestep.

        Args:
            action: Control vector in ``[-1, 1]^action_dim``.

        Returns:
            Tuple of ``(obs, reward, terminated, truncated, info)``.
        """
        self._step_count += 1
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        reward = 0.0
        terminated = False
        truncated = self._step_count >= self.max_steps
        info: dict[str, Any] = {}
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Render the current environment state.

        Returns:
            RGB array when ``render_mode="rgb_array"``, otherwise ``None``.
        """
        raise NotImplementedError("Rendering not yet implemented.")

    def close(self) -> None:
        """Release any resources held by the environment."""
