"""Interactive 3D trajectory navigator for exploring recorded flight paths."""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Interactively navigate 3D trajectory data.")
    parser.add_argument(
        "--trajectory-dir",
        type=str,
        required=True,
        help="Directory containing trajectory files.",
    )
    args = parser.parse_args()

    print(f"[view_3d_navigator] trajectory_dir={args.trajectory_dir}")
    # TODO: build interactive matplotlib or plotly viewer for trajectories in the directory


if __name__ == "__main__":
    main()
