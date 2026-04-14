"""Generate comparison plots against survey/literature baselines."""
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Generate baseline comparison plots.")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory containing results data.")
    parser.add_argument("--output", type=str, required=True, help="Output file path for comparison plot.")
    args = parser.parse_args()

    print(f"[generate_survey_baselines] results_dir={args.results_dir}")
    print(f"[generate_survey_baselines] output={args.output}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    # TODO: load baseline data from results_dir, render comparison chart


if __name__ == "__main__":
    main()
