"""Plot results from shadow (A/B) testing between two agent policies."""
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Plot shadow/A-B testing results.")
    parser.add_argument("--shadow-results", type=str, required=True, help="Path to shadow results file.")
    parser.add_argument("--output", type=str, required=True, help="Output image file path.")
    args = parser.parse_args()

    print(f"[plot_shadow_results] shadow_results={args.shadow_results}")
    print(f"[plot_shadow_results] output={args.output}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    # TODO: load shadow results, generate side-by-side comparison plot


if __name__ == "__main__":
    main()
