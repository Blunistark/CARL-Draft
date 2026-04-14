"""DRL agent evaluation utilities."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from carl.agents.base_agent import BaseAgent


def evaluate_agent(
    agent: "BaseAgent",
    env: Any,
    n_episodes: int = 10,
    render: bool = False,
) -> dict[str, float]:
    """Run ``agent`` in ``env`` for ``n_episodes`` and return summary metrics.

    Each episode is rolled out deterministically (no exploration noise).
    The function aggregates per-episode returns, episode lengths, and success
    flags into a single metrics dict.

    Args:
        agent: A trained agent implementing
            :meth:`~carl.agents.BaseAgent.select_action`.
        env: A Gym-compatible environment with ``reset`` / ``step`` API.
        n_episodes: Number of evaluation episodes to run.
        render: If ``True``, call ``env.render()`` at each step.

    Returns:
        Dict with keys:
            - ``mean_return``: Mean episodic undiscounted return.
            - ``std_return``: Standard deviation of episodic returns.
            - ``mean_length``: Mean episode length in steps.
            - ``success_rate``: Fraction of episodes flagged as successful.
    """
    returns: list[float] = []
    lengths: list[int] = []
    successes: list[bool] = []

    for _ in range(n_episodes):
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        done = False
        ep_return = 0.0
        ep_length = 0

        while not done:
            action = agent.select_action(obs)
            step_result = env.step(action)

            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result  # type: ignore[misc]

            ep_return += float(reward)
            ep_length += 1

            if render:
                env.render()

        returns.append(ep_return)
        lengths.append(ep_length)
        successes.append(bool(info.get("success", False)))

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_length": float(np.mean(lengths)),
        "success_rate": float(np.mean(successes)),
    }
