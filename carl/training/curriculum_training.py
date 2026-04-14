"""Training loop with adaptive curriculum management."""
from __future__ import annotations

from typing import Any


def train_with_curriculum(config: dict[str, Any]) -> dict[str, Any]:
    """Train an agent using an adaptive :class:`~carl.curriculum.CurriculumManager`.

    The curriculum manager dynamically controls environment difficulty based
    on recent agent performance.  Training checkpoints and curriculum state
    are saved together so that training can be resumed from any stage.

    Args:
        config: Training configuration dict.  Expected keys include all keys
            from :func:`~carl.training.standard.train_standard` plus:

            - ``stages`` (list[dict]): Ordered curriculum stage configs.
            - ``promotion_threshold`` (float): Success fraction to advance.
            - ``window_size`` (int): Episode window for promotion check.
            - ``save_curriculum_state`` (bool): Whether to persist the
              curriculum manager alongside agent checkpoints.

    Returns:
        Dict of training metrics including per-stage statistics and final
        evaluation return.
    """
    # --- Curriculum setup ---
    # curriculum = CurriculumManager(
    #     stages=config["stages"],
    #     promotion_threshold=config.get("promotion_threshold", 0.7),
    #     window_size=config.get("window_size", 100),
    # )

    # --- Training loop ---
    # for episode in range(config["max_episodes"]):
    #     stage_config = curriculum.get_stage()
    #     env = _make_env(stage_config)
    #     result = _run_episode(agent, env)
    #     curriculum.record_episode(result["success"])
    #     if config.get("save_curriculum_state"):
    #         _save_curriculum(curriculum, config["checkpoint_dir"])

    raise NotImplementedError("train_with_curriculum is not yet implemented.")
