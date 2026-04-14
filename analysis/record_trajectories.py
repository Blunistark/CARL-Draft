"""Record multiple trajectory rollouts for offline analysis."""
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Record multiple rollout trajectories.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--n-episodes", type=int, default=20, help="Number of episodes to record.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save trajectory files.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[record_trajectories] checkpoint={args.checkpoint}")
    print(f"[record_trajectories] n_episodes={args.n_episodes}")
    print(f"[record_trajectories] output_dir={args.output_dir}")
    # TODO: load agent, roll out episodes, save each trajectory as CSV


if __name__ == "__main__":
    main()
