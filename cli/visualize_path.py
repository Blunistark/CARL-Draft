"""CLI script for visualizing flight paths from trajectory files."""
import argparse

import numpy as np

from carl.logging.spatial_logger import SpatialLogger
from carl.viz.path_viewer import PathViewer


def main():
    parser = argparse.ArgumentParser(description="Visualize a UAV flight path.")
    parser.add_argument("--trajectory-file", type=str, required=True, help="Path to trajectory file (.json or .npz).")
    parser.add_argument("--save", type=str, default=None, help="Path to save the rendered image.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open interactive viewer window (requires a display).",
    )
    args = parser.parse_args()

    logger = SpatialLogger.load(args.trajectory_file)
    positions = np.array([step["state"][:3] for step in logger._steps], dtype=np.float32)

    viewer = PathViewer()
    viewer.add_trajectory(positions=positions, label="agent")
    viewer.render(save_path=args.save)

    if args.interactive:
        try:
            import matplotlib.pyplot as plt
            plt.show()
        except ImportError:
            print("matplotlib is required for interactive mode.")


if __name__ == "__main__":
    main()
