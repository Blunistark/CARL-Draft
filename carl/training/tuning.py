"""Fine-tuning utilities for trained CARL agents."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def fine_tune_smoothness(
    agent_path: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Fine-tune a trained agent to produce smoother control outputs.

    Loads a checkpoint from ``agent_path`` and continues training with a
    reward signal that emphasises action smoothness (increased
    ``smoothness`` weight in :class:`~carl.reward.RewardEngine`) while
    keeping the tracking objective active.

    This procedure is typically applied after the main training run to
    reduce actuator wear and improve passenger / airframe comfort in
    real-world deployment.

    Args:
        agent_path: Path to the agent checkpoint directory produced by a
            prior :func:`~carl.training.standard.train_standard` or
            :func:`~carl.training.staged.train_staged` run.
        config: Fine-tuning configuration dict.  Expected keys include:

            - ``env_id`` (str): Environment to fine-tune on.
            - ``fine_tune_steps`` (int): Number of additional training steps.
            - ``smoothness_weight`` (float): Overridden smoothness reward weight.
            - ``tracking_weight`` (float): Tracking reward weight during FT.
            - ``actor_lr`` (float): Optional reduced actor learning rate.
            - ``output_dir`` (str | Path): Where to save the fine-tuned agent.

    Returns:
        Dict of fine-tuning metrics (``steps_completed``, ``final_return``,
        ``mean_action_delta``).
    """
    # --- Load pre-trained agent ---
    # agent = PioneerAgentV1.load(agent_path)

    # --- Reconfigure reward engine ---
    # reward_engine = RewardEngine(weights={
    #     "smoothness": config["smoothness_weight"],
    #     "tracking": config.get("tracking_weight", 1.0),
    # })

    # --- Fine-tune loop ---
    # for step in range(config["fine_tune_steps"]):
    #     ...

    raise NotImplementedError("fine_tune_smoothness is not yet implemented.")
