"""Interactive visualisation of curriculum stages and agent progress."""
from __future__ import annotations

from typing import Any


class CurriculumViewer:
    """Interactive dashboard for exploring curriculum stage progression.

    Renders live or post-hoc plots of:
      - Stage index over episodes.
      - Per-stage episode count and success distribution.
      - Reward component breakdown across stages.

    Args:
        figsize: Figure dimensions ``(width, height)`` in inches.
        style: Matplotlib style sheet name (e.g. ``"seaborn-v0_8"``).
    """

    def __init__(
        self,
        figsize: tuple[float, float] = (12, 6),
        style: str = "default",
    ) -> None:
        self.figsize = figsize
        self.style = style
        self._history: list[dict[str, Any]] = []

    def update(self, record: dict[str, Any]) -> None:
        """Append one episode record to the viewer's history buffer.

        Args:
            record: Dict with at minimum ``episode``, ``stage``, ``success``,
                and ``total_reward`` keys.
        """
        self._history.append(record)

    def render(self, save_path: str | None = None) -> None:
        """Draw the curriculum dashboard.

        Args:
            save_path: If provided, save the figure to this path instead of
                displaying it interactively.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for CurriculumViewer. "
                "Install with: pip install matplotlib"
            ) from exc

        from carl.curriculum.visualize_curriculum import plot_curriculum_progress

        plot_curriculum_progress(self._history, save_path=save_path)

    def clear(self) -> None:
        """Clear the accumulated history buffer."""
        self._history = []
