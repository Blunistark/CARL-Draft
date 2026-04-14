"""Utilities for visualising curriculum progression over training."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_curriculum_progress(
    history: list[dict[str, Any]],
    save_path: str | Path | None = None,
) -> None:
    """Plot curriculum stage transitions and success rates over time.

    Generates a figure with two panels:
      1. Stage index vs. episode number.
      2. Rolling success rate vs. episode number, with stage boundaries
         annotated as vertical dashed lines.

    Args:
        history: List of per-episode dicts, each containing at least the
            keys ``episode``, ``stage``, and ``success`` (bool).
        save_path: If provided, the figure is saved to this path instead of
            being displayed interactively.  Supports any format understood by
            ``matplotlib`` (e.g. ``".png"``, ``".pdf"``).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for curriculum visualisation. "
            "Install it with: pip install matplotlib"
        ) from exc

    episodes = [h["episode"] for h in history]
    stages = [h["stage"] for h in history]
    successes = [int(h["success"]) for h in history]

    window = 50
    rolling_success = [
        sum(successes[max(0, i - window) : i + 1])
        / len(successes[max(0, i - window) : i + 1])
        for i in range(len(successes))
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.step(episodes, stages, where="post", color="steelblue")
    ax1.set_ylabel("Curriculum Stage")
    ax1.set_title("Curriculum Progression")

    ax2.plot(episodes, rolling_success, color="darkorange", label=f"Rolling {window}-ep")
    ax2.set_ylabel("Success Rate")
    ax2.set_xlabel("Episode")
    ax2.set_ylim(0, 1)
    ax2.legend()

    # Annotate stage transition boundaries
    stage_changes = [
        episodes[i]
        for i in range(1, len(stages))
        if stages[i] != stages[i - 1]
    ]
    for ep in stage_changes:
        ax1.axvline(ep, color="grey", linestyle="--", linewidth=0.8)
        ax2.axvline(ep, color="grey", linestyle="--", linewidth=0.8)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    else:
        plt.show()

    plt.close(fig)
