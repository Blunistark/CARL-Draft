"""Policy interrogation: feature importance, sensitivity analysis, action explanation."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from carl.agents.base_agent import BaseAgent


class PolicyInterrogator:
    """Interrogates a trained policy to understand its decision-making.

    Provides tools for:
      - Explaining individual action selections.
      - Measuring sensitivity of actions to each observation dimension.
      - Probing decision boundaries in observation space.

    Args:
        agent: A trained agent whose :meth:`~carl.agents.BaseAgent.select_action`
            method will be interrogated.
        perturbation_scale: Standard deviation of Gaussian perturbations used
            during sensitivity analysis.
        n_samples: Number of perturbation samples per sensitivity estimate.
    """

    def __init__(
        self,
        agent: "BaseAgent",
        perturbation_scale: float = 0.1,
        n_samples: int = 100,
    ) -> None:
        self.agent = agent
        self.perturbation_scale = perturbation_scale
        self.n_samples = n_samples

    def explain_action(self, state: np.ndarray) -> dict[str, Any]:
        """Generate a feature-importance explanation for the action at ``state``.

        Uses finite-difference perturbations to estimate how much each
        observation dimension contributes to the selected action.

        Args:
            state: Flat observation array for which to explain the action.

        Returns:
            Dict with keys:
              - ``action``: The action selected at ``state``.
              - ``feature_importance``: Array of absolute sensitivities,
                one per observation dimension.
        """
        base_action = self.agent.select_action(state)
        importance = np.zeros(len(state), dtype=np.float32)

        for i in range(len(state)):
            perturbed = state.copy()
            perturbed[i] += self.perturbation_scale
            action_plus = self.agent.select_action(perturbed)
            perturbed[i] -= 2 * self.perturbation_scale
            action_minus = self.agent.select_action(perturbed)
            importance[i] = float(
                np.mean(np.abs(np.asarray(action_plus) - np.asarray(action_minus)))
            ) / (2 * self.perturbation_scale)

        return {"action": base_action, "feature_importance": importance}

    def sensitivity_analysis(self, env: Any) -> dict[str, Any]:
        """Estimate per-feature action sensitivity over a set of environment states.

        Resets ``env`` and collects ``n_samples`` states by rolling out the
        policy, then calls :meth:`explain_action` on each and averages the
        resulting importance scores.

        Args:
            env: Gym-compatible environment used to collect representative states.

        Returns:
            Dict with key ``mean_importance`` (array of length
            ``observation_dim``) and ``std_importance``.
        """
        importances: list[np.ndarray] = []
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result

        for _ in range(self.n_samples):
            action = self.agent.select_action(obs)
            expl = self.explain_action(np.asarray(obs, dtype=np.float32))
            importances.append(expl["feature_importance"])
            step_result = env.step(action)
            obs = step_result[0] if len(step_result) >= 4 else step_result[0]
            done = step_result[2] if len(step_result) == 4 else (
                step_result[2] or step_result[3]
            )
            if done:
                result = env.reset()
                obs = result[0] if isinstance(result, tuple) else result

        importance_arr = np.stack(importances)
        return {
            "mean_importance": importance_arr.mean(axis=0),
            "std_importance": importance_arr.std(axis=0),
        }
