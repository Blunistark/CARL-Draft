"""Standard TD3 training loop (no curriculum)."""
from __future__ import annotations

from typing import Any


def train_standard(config: dict[str, Any]) -> dict[str, Any]:
    """Run a standard TD3 training loop using the provided configuration.

    Training proceeds for ``config["total_steps"]`` environment interactions,
    collecting experience into a replay buffer and updating the agent every
    ``config.get("update_every", 1)`` steps.  Checkpoints are saved
    periodically to ``config["checkpoint_dir"]``.

    Args:
        config: Training configuration dict.  Expected keys include:

            - ``env_id`` (str): Gym environment identifier or class.
            - ``total_steps`` (int): Total environment steps to train for.
            - ``batch_size`` (int): Replay buffer sample batch size.
            - ``replay_buffer_size`` (int): Maximum replay buffer capacity.
            - ``start_steps`` (int): Random exploration steps before training.
            - ``update_every`` (int): Agent update frequency (steps).
            - ``checkpoint_dir`` (str | Path): Directory for checkpoints.
            - ``log_interval`` (int): Steps between console log lines.
            - ``eval_interval`` (int): Steps between evaluation episodes.
            - ``n_eval_episodes`` (int): Episodes per evaluation run.
            - ``seed`` (int): Global random seed.

    Returns:
        Dict of final training metrics (``total_steps``, ``final_return``,
        ``best_return``).
    """
    # --- Setup ---
    # env = _make_env(config)
    # agent = _build_agent(config)
    # replay_buffer = ReplayBuffer(config["replay_buffer_size"])

    # --- Main training loop ---
    # for step in range(config["total_steps"]):
    #     obs, action, reward, next_obs, done = _collect_transition(env, agent, step, config)
    #     replay_buffer.add(obs, action, reward, next_obs, done)
    #     if step >= config["start_steps"] and step % config.get("update_every", 1) == 0:
    #         batch = replay_buffer.sample(config["batch_size"])
    #         agent.update(batch)
    #     if step % config["log_interval"] == 0:
    #         _log(step, ...)
    #     if step % config["eval_interval"] == 0:
    #         _evaluate_and_checkpoint(agent, env, config, step)

    raise NotImplementedError("train_standard is not yet implemented.")


def train_standard_lstm(config: dict[str, Any]) -> dict[str, Any]:
    """Run a standard TD3 training loop with an LSTM-based policy.

    Drop-in variant of :func:`train_standard` that substitutes an LSTM actor
    and critic for the default MLP networks.

    Args:
        config: Training configuration dict (same schema as
            :func:`train_standard`, plus optional ``lstm_hidden_size`` and
            ``lstm_num_layers`` keys).

    Returns:
        Dict of final training metrics.
    """
    raise NotImplementedError("train_standard_lstm is not yet implemented.")


def train_standard_transformer(config: dict[str, Any]) -> dict[str, Any]:
    """Run a standard TD3 training loop with a Transformer-based policy.

    Drop-in variant of :func:`train_standard` that substitutes a Transformer
    actor and critic for the default MLP networks.

    Args:
        config: Training configuration dict (same schema as
            :func:`train_standard`, plus optional ``transformer_nhead``,
            ``transformer_num_layers``, and ``transformer_dim_feedforward``
            keys).

    Returns:
        Dict of final training metrics.
    """
    raise NotImplementedError("train_standard_transformer is not yet implemented.")
