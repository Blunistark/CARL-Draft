"""CLI script for visualizing failure cases from a training run."""
import argparse
import json
import os


def _load_episode_results(run_dir: str) -> list[dict]:
    """Load per-episode results from the run directory."""
    results_path = os.path.join(run_dir, "episode_results.json")
    if not os.path.isfile(results_path):
        return []
    with open(results_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Visualize the worst (failed) episodes from a training run.")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to the training run directory.")
    parser.add_argument("--n-worst", type=int, default=5, help="Number of worst episodes to highlight.")
    args = parser.parse_args()

    episodes = _load_episode_results(args.run_dir)
    if not episodes:
        print("No episode results found in run directory.")
        return

    sorted_eps = sorted(episodes, key=lambda e: e.get("reward", float("inf")))
    worst = sorted_eps[: args.n_worst]
    print(f"Top {args.n_worst} worst episodes:")
    for i, ep in enumerate(worst, 1):
        print(f"  {i}. Episode {ep.get('episode_id', '?')} | reward={ep.get('reward', '?')}")


if __name__ == "__main__":
    main()
