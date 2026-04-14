"""Record the smoothest agent rollout as a video or GIF."""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Record the smoothest agent rollout.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--env-config", type=str, required=True, help="Path to environment config.")
    parser.add_argument("--output", type=str, required=True, help="Output video/GIF file path.")
    args = parser.parse_args()

    print(f"[record_smooth_champion] checkpoint={args.checkpoint}")
    print(f"[record_smooth_champion] env_config={args.env_config}")
    print(f"[record_smooth_champion] output={args.output}")
    # TODO: run multiple rollouts, pick smoothest by action-variation metric, save recording


if __name__ == "__main__":
    main()
