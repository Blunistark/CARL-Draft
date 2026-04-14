"""Neural network architectures for CARL agents (MLP, LSTM, Transformer)."""
from __future__ import annotations

from typing import Any

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    nn = None  # type: ignore[assignment]


def _require_torch() -> None:
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for neural network architectures. "
            "Install it with: pip install torch"
        )


class ActorNetwork(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """Standard MLP actor network for TD3.

    Maps observations to deterministic actions in ``[-1, 1]^action_dim``
    via a ``tanh`` output activation.

    Args:
        state_dim: Input observation dimensionality.
        action_dim: Output action dimensionality.
        hidden_dim: Width of each hidden layer.
        n_layers: Number of hidden layers.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
    ) -> None:
        _require_torch()
        super().__init__()
        import torch.nn as _nn
        layers: list[Any] = [_nn.Linear(state_dim, hidden_dim), _nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [_nn.Linear(hidden_dim, hidden_dim), _nn.ReLU()]
        layers.append(_nn.Linear(hidden_dim, action_dim))
        layers.append(_nn.Tanh())
        self.net = _nn.Sequential(*layers)

    def forward(self, state: "torch.Tensor") -> "torch.Tensor":
        """Forward pass.

        Args:
            state: Batch of observations, shape ``(batch, state_dim)``.

        Returns:
            Actions tensor, shape ``(batch, action_dim)`` in ``[-1, 1]``.
        """
        return self.net(state)


class CriticNetwork(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """Twin MLP critic network for TD3.

    Implements two independent Q-value heads to reduce overestimation bias.

    Args:
        state_dim: Observation dimensionality.
        action_dim: Action dimensionality.
        hidden_dim: Width of each hidden layer.
        n_layers: Number of hidden layers per Q-head.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
    ) -> None:
        _require_torch()
        super().__init__()
        import torch.nn as _nn

        def _build_head() -> "_nn.Sequential":
            layers: list[Any] = [_nn.Linear(state_dim + action_dim, hidden_dim), _nn.ReLU()]
            for _ in range(n_layers - 1):
                layers += [_nn.Linear(hidden_dim, hidden_dim), _nn.ReLU()]
            layers.append(_nn.Linear(hidden_dim, 1))
            return _nn.Sequential(*layers)

        self.q1 = _build_head()
        self.q2 = _build_head()

    def forward(
        self, state: "torch.Tensor", action: "torch.Tensor"
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """Forward pass returning both Q-value estimates.

        Args:
            state: Batch of observations, shape ``(batch, state_dim)``.
            action: Batch of actions, shape ``(batch, action_dim)``.

        Returns:
            Tuple ``(q1, q2)`` each of shape ``(batch, 1)``.
        """
        import torch as _torch
        sa = _torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


class LSTMActor(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """LSTM-based actor for partially observable / sequential observation settings.

    Maintains a hidden state across timesteps, allowing the agent to condition
    actions on observation history rather than a single Markovian state.

    Args:
        state_dim: Dimensionality of each observation in the sequence.
        action_dim: Output action dimensionality.
        hidden_dim: LSTM hidden state size and MLP head width.
        num_layers: Number of stacked LSTM layers.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 1,
    ) -> None:
        _require_torch()
        super().__init__()
        import torch.nn as _nn
        self.lstm = _nn.LSTM(state_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = _nn.Sequential(
            _nn.Linear(hidden_dim, hidden_dim),
            _nn.ReLU(),
            _nn.Linear(hidden_dim, action_dim),
            _nn.Tanh(),
        )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(
        self,
        state_seq: "torch.Tensor",
        hidden: "tuple[torch.Tensor, torch.Tensor] | None" = None,
    ) -> "tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]":
        """Forward pass over a sequence of observations.

        Args:
            state_seq: Observation sequence, shape ``(batch, seq_len, state_dim)``.
            hidden: Optional initial LSTM hidden state ``(h, c)``.

        Returns:
            Tuple of ``(actions, (h_n, c_n))`` where ``actions`` has shape
            ``(batch, seq_len, action_dim)``.
        """
        out, hidden_out = self.lstm(state_seq, hidden)
        actions = self.head(out)
        return actions, hidden_out


class TransformerActor(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """Transformer-based actor for sequence-to-action mapping.

    Uses a causal self-attention encoder to aggregate observation history
    and produce actions for each timestep in the sequence.

    Args:
        state_dim: Dimensionality of each observation token.
        action_dim: Output action dimensionality.
        d_model: Internal embedding / attention dimension.
        nhead: Number of attention heads.
        num_encoder_layers: Depth of the Transformer encoder.
        dim_feedforward: Width of the FFN sublayer.
        dropout: Dropout probability.
        max_seq_len: Maximum sequence length (used for positional encoding).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 256,
    ) -> None:
        _require_torch()
        super().__init__()
        import torch
        import torch.nn as _nn
        self.input_proj = _nn.Linear(state_dim, d_model)
        encoder_layer = _nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = _nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.head = _nn.Sequential(
            _nn.Linear(d_model, d_model),
            _nn.ReLU(),
            _nn.Linear(d_model, action_dim),
            _nn.Tanh(),
        )
        # Learnable positional encoding
        self.pos_embedding = _nn.Parameter(torch.zeros(1, max_seq_len, d_model))

    def forward(self, state_seq: "torch.Tensor") -> "torch.Tensor":
        """Forward pass over a sequence of observations.

        Args:
            state_seq: Observation sequence, shape ``(batch, seq_len, state_dim)``.

        Returns:
            Actions tensor, shape ``(batch, seq_len, action_dim)``.
        """
        x = self.input_proj(state_seq)
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.encoder(x)
        return self.head(x)
