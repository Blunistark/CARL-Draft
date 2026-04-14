"""Generate publication-quality plots for the CARL paper."""
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Generate paper-quality plots.")
    parser.add_argument("--results-dir", type=str, required=True, help="Directory containing results data.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write output figures.")
    parser.add_argument(
        "--format",
        type=str,
        choices=["pdf", "png"],
        default="pdf",
        help="Output figure format.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[generate_paper_plots] results_dir={args.results_dir}")
    print(f"[generate_paper_plots] output_dir={args.output_dir} format={args.format}")
    # TODO: load results, generate publication-quality matplotlib figures


if __name__ == "__main__":
    main()
