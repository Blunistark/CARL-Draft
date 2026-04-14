"""CLI script that exports trajectory data to CSV or JSON."""
import argparse
import csv
import json
import os

from carl.logging.spatial_logger import SpatialLogger


def _export_json(logger: SpatialLogger, output_path: str) -> None:
    payload = {"episode_id": logger.episode_id, "steps": logger._steps}
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


def _export_csv(logger: SpatialLogger, output_path: str) -> None:
    if not logger._steps:
        print("No steps to export.")
        return
    fieldnames = list(logger._steps[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(logger._steps)


def main():
    parser = argparse.ArgumentParser(description="Export trajectory data from a saved log file.")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to a saved SpatialLogger file (.json or .npz).")
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["csv", "json"],
        default="csv",
        help="Output format for trajectory data.",
    )
    parser.add_argument("--output", type=str, required=True, help="Output file path.")
    args = parser.parse_args()

    logger = SpatialLogger.load(args.run_dir)
    if args.output_format == "json":
        _export_json(logger, args.output)
    else:
        _export_csv(logger, args.output)
    print(f"Trajectory exported to {args.output} ({args.output_format})")


if __name__ == "__main__":
    main()
