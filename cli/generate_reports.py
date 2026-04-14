"""CLI script that generates summary training reports from logged data."""
import argparse
import json
import os


def _collect_stats(run_dir: str) -> dict:
    """Collect basic summary statistics from a run directory."""
    stats: dict = {"run_dir": run_dir, "episodes": None, "mean_reward": None}
    metrics_path = os.path.join(run_dir, "metrics.json")
    if os.path.isfile(metrics_path):
        with open(metrics_path) as f:
            data = json.load(f)
        stats.update(data)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Generate training summary reports.")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to the training run directory.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write report files.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    stats = _collect_stats(args.run_dir)

    report_path = os.path.join(args.output_dir, "summary.json")
    with open(report_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
