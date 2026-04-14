"""Curriculum manager for held-out / unseen evaluation scenarios."""
from __future__ import annotations

from typing import Any


class UnseenCurriculumManager:
    """Manages a pool of held-out scenarios for zero-shot generalisation tests.

    Unseen scenarios are never used during training; they are sampled only
    at evaluation time to measure out-of-distribution agent performance.

    Args:
        scenarios: List of scenario configuration dicts to hold out.
        seed: Optional random seed for reproducible sampling order.
    """

    def __init__(
        self,
        scenarios: list[dict[str, Any]],
        seed: int | None = None,
    ) -> None:
        self.scenarios = scenarios
        self._results: list[dict[str, Any]] = []
        import random as _random
        self._rng = _random.Random(seed)

    def sample_scenario(self) -> dict[str, Any]:
        """Randomly sample one held-out scenario configuration.

        Returns:
            A scenario configuration dict from the held-out pool.

        Raises:
            ValueError: If the scenario pool is empty.
        """
        if not self.scenarios:
            raise ValueError("No unseen scenarios registered.")
        return self._rng.choice(self.scenarios)

    def register_result(self, scenario: dict[str, Any], metrics: dict[str, Any]) -> None:
        """Record evaluation metrics for a completed unseen scenario.

        Args:
            scenario: The scenario configuration that was evaluated.
            metrics: Dict of performance metrics (e.g. success, tracking_error).
        """
        self._results.append({"scenario": scenario, "metrics": metrics})

    def summary(self) -> dict[str, Any]:
        """Aggregate summary statistics over all registered results.

        Returns:
            Dict containing ``num_evaluated`` and per-metric mean values.
        """
        if not self._results:
            return {"num_evaluated": 0}

        all_keys = {k for r in self._results for k in r["metrics"]}
        summary: dict[str, Any] = {"num_evaluated": len(self._results)}
        for key in all_keys:
            vals = [
                r["metrics"][key]
                for r in self._results
                if key in r["metrics"]
            ]
            summary[f"mean_{key}"] = sum(vals) / len(vals)
        return summary
