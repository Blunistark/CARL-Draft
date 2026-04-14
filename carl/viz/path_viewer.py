"""Interactive 3-D visualisation of fixed-wing UAV flight trajectories."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class PathViewer:
    """Renders 3-D flight trajectories for fixed-wing UAV episodes.

    Overlays the reference path (waypoints) alongside the agent's actual
    flown trajectory.  Optionally colours the trajectory by speed,
    cross-track error, or reward magnitude.

    Args:
        figsize: Figure dimensions ``(width, height)`` in inches.
        elev: Initial elevation angle (degrees) for the 3-D axes.
        azim: Initial azimuth angle (degrees) for the 3-D axes.
    """

    def __init__(
        self,
        figsize: tuple[float, float] = (10, 8),
        elev: float = 25.0,
        azim: float = -60.0,
    ) -> None:
        self.figsize = figsize
        self.elev = elev
        self.azim = azim
        self._trajectories: list[dict[str, Any]] = []
        self._waypoints: list[np.ndarray] = []

    def add_trajectory(
        self,
        positions: np.ndarray,
        label: str = "agent",
        color: str | None = None,
        scalar: np.ndarray | None = None,
    ) -> None:
        """Register a flight trajectory for display.

        Args:
            positions: Array of shape ``(T, 3)`` containing ``[x, y, z]``
                positions at each timestep.
            label: Legend label for this trajectory.
            color: Matplotlib colour string.  If ``None``, auto-assigned.
            scalar: Optional per-step scalar (e.g. speed, reward) used to
                colour the trajectory line.  Shape ``(T,)``.
        """
        self._trajectories.append(
            {
                "positions": np.asarray(positions, dtype=np.float32),
                "label": label,
                "color": color,
                "scalar": scalar,
            }
        )

    def set_waypoints(self, waypoints: np.ndarray) -> None:
        """Set the reference waypoint path to overlay on the plot.

        Args:
            waypoints: Array of shape ``(N, 3)`` containing waypoint positions.
        """
        self._waypoints = [np.asarray(wp, dtype=np.float32) for wp in waypoints]

    def render(self, save_path: str | Path | None = None) -> None:
        """Render the 3-D trajectory plot.

        Args:
            save_path: If provided, save the figure to this path instead of
                displaying it interactively.
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for PathViewer. "
                "Install with: pip install matplotlib"
            ) from exc

        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=self.elev, azim=self.azim)

        # Draw waypoints
        if self._waypoints:
            wp_arr = np.stack(self._waypoints)
            ax.plot(
                wp_arr[:, 0], wp_arr[:, 1], wp_arr[:, 2],
                "k--", linewidth=1.5, label="reference", alpha=0.7,
            )
            ax.scatter(
                wp_arr[:, 0], wp_arr[:, 1], wp_arr[:, 2],
                color="black", s=30, zorder=5,
            )

        # Draw trajectories
        for traj in self._trajectories:
            pos = traj["positions"]
            ax.plot(
                pos[:, 0], pos[:, 1], pos[:, 2],
                label=traj["label"],
                color=traj["color"],
                linewidth=1.5,
            )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("UAV Flight Trajectory")
        ax.legend()
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=150)
        else:
            plt.show()

        plt.close(fig)

    def clear(self) -> None:
        """Reset stored trajectories and waypoints."""
        self._trajectories = []
        self._waypoints = []
