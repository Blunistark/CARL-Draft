"""Compare action smoothness metrics across multiple training runs."""
import argparse
import json
import os


def _smoothness_score(run_dir: str) -> float | None:
    """Load smoothness score from a run's metrics file, if present."""
    path = os.path.join(run_dir, "metrics.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("smoothness")


def main():
    parser = argparse.ArgumentParser(description="Compare action smoothness across runs.")
    parser.add_argument("--run-dirs", type=str, nargs="+", required=True, help="Paths to run directories.")
    parser.add_argument("--output", type=str, default=None, help="Path to save comparison plot or report.")
    args = parser.parse_args()

    results = {}
    for run_dir in args.run_dirs:
        score = _smoothness_score(run_dir)
        results[run_dir] = score
        label = f"{score:.4f}" if score is not None else "N/A"
        print(f"  {run_dir}: smoothness={label}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
    # TODO: generate matplotlib comparison plot


if __name__ == "__main__":
    main()
