"""Staged training with progressive difficulty levels."""
from __future__ import annotations

from typing import Any


def train_staged(config: dict[str, Any]) -> dict[str, Any]:
    """Run a staged TD3 training loop with difficulty progression.

    Wraps :func:`~carl.training.standard.train_standard` with a
    :class:`~carl.curriculum.CurriculumManager` that advances the environment
    difficulty when the agent achieves a sufficient success rate on the
    current stage.

    Args:
        config: Training configuration dict.  In addition to the keys
            documented in :func:`~carl.training.standard.train_standard`,
            expects:

            - ``stages`` (list[dict]): Ordered list of stage configs.
            - ``promotion_threshold`` (float): Success rate to advance stage.
            - ``window_size`` (int): Episodes considered for promotion check.

    Returns:
        Dict of training metrics including per-stage episode counts and
        final return.
    """
    raise NotImplementedError("train_staged is not yet implemented.")


def train_staged_lstm(config: dict[str, Any]) -> dict[str, Any]:
    """Staged training using an LSTM-based actor.

    Identical to :func:`train_staged` but substitutes
    :class:`~carl.models.LSTMActor` for the default MLP actor.

    Args:
        config: Training configuration dict (same schema as
            :func:`train_staged` with an additional ``lstm_hidden_dim`` key).

    Returns:
        Dict of training metrics.
    """
    raise NotImplementedError("train_staged_lstm is not yet implemented.")


def train_staged_transformer(config: dict[str, Any]) -> dict[str, Any]:
    """Staged training using a Transformer-based actor.

    Identical to :func:`train_staged` but substitutes
    :class:`~carl.models.TransformerActor` for the default MLP actor.

    Args:
        config: Training configuration dict (same schema as
            :func:`train_staged` with additional Transformer hyper-parameter
            keys: ``d_model``, ``nhead``, ``num_encoder_layers``).

    Returns:
        Dict of training metrics.
    """
    raise NotImplementedError("train_staged_transformer is not yet implemented.")
