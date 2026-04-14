"""Shadow comparison testing between two agent policies."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from carl.agents.base_agent import BaseAgent


class ShadowTester:
    """Runs paired shadow-mode comparison tests between two agents.

    Both agents are evaluated on identical episode seeds so that
    environmental stochasticity is controlled.  The resulting per-episode
    metrics are stored and can be summarised via :meth:`report`.

    Args:
        n_episodes: Number of comparison episodes per call to :meth:`compare`.
        seeds: Optional list of explicit seeds for environment resets.
            If ``None``, sequential integers starting from 0 are used.
    """

    def __init__(
        self,
        n_episodes: int = 20,
        seeds: list[int] | None = None,
    ) -> None:
        self.n_episodes = n_episodes
        self.seeds = seeds or list(range(n_episodes))
        self._results: list[dict[str, Any]] = []

    def compare(
        self,
        agent_a: "BaseAgent",
        agent_b: "BaseAgent",
        env: Any,
    ) -> list[dict[str, Any]]:
        """Run both agents on the same set of episodes and record results.

        Args:
            agent_a: First agent (e.g. candidate / new policy).
            agent_b: Second agent (e.g. baseline / reference policy).
            env: Gym-compatible environment.

        Returns:
            List of per-episode result dicts, each containing ``seed``,
            ``return_a``, ``return_b``, and ``delta`` (a - b).
        """
        from carl.evaluation.eval_drl import evaluate_agent

        episode_results: list[dict[str, Any]] = []
        for seed in self.seeds[: self.n_episodes]:
            # Evaluate agent_a
            env.reset(seed=seed)
            metrics_a = evaluate_agent(agent_a, env, n_episodes=1)
            # Evaluate agent_b on same seed
            env.reset(seed=seed)
            metrics_b = evaluate_agent(agent_b, env, n_episodes=1)

            record = {
                "seed": seed,
                "return_a": metrics_a["mean_return"],
                "return_b": metrics_b["mean_return"],
                "delta": metrics_a["mean_return"] - metrics_b["mean_return"],
            }
            episode_results.append(record)

        self._results.extend(episode_results)
        return episode_results

    def report(self) -> dict[str, Any]:
        """Summarise all comparison results collected so far.

        Returns:
            Dict containing ``num_episodes``, ``mean_delta``,
            ``win_rate_a`` (fraction where agent_a > agent_b), and
            ``win_rate_b``.
        """
        if not self._results:
            return {"num_episodes": 0}

        deltas = [r["delta"] for r in self._results]
        wins_a = sum(1 for d in deltas if d > 0)
        n = len(deltas)
        return {
            "num_episodes": n,
            "mean_delta": sum(deltas) / n,
            "win_rate_a": wins_a / n,
            "win_rate_b": (n - wins_a) / n,
        }
