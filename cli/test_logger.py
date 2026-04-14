"""Manual demo / integration test of the SpatialLogger.

Not a unit test — run directly to verify the logger works end-to-end.
"""
import os

import numpy as np

from carl.logging.spatial_logger import SpatialLogger


def main():
    run_dir = os.path.join(os.getcwd(), "runs", "logger_demo")
    os.makedirs(run_dir, exist_ok=True)

    logger = SpatialLogger(episode_id="demo_episode")
    print(f"SpatialLogger initialised (episode_id={logger.episode_id})")

    # Log a few synthetic waypoints.
    for step in range(5):
        state = np.array([float(step), 0.0, 100.0, 0.0, 0.0, 0.0])
        action = np.array([0.0, 0.0, 0.0, 0.5])
        logger.log_step(state=state, action=action, reward=float(step))

    save_path = os.path.join(run_dir, "demo_trajectory.json")
    logger.save(save_path)
    print(f"Demo complete. Log written to: {save_path} ({len(logger)} steps)")


if __name__ == "__main__":
    main()
