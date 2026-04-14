"""Generate SHAP summary plots for a trained CARL agent."""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Generate SHAP summary plots.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--data", type=str, required=True, help="Path to observation data file.")
    parser.add_argument("--output", type=str, required=True, help="Output image file path.")
    args = parser.parse_args()

    print(f"[generate_shap_plots] checkpoint={args.checkpoint}")
    print(f"[generate_shap_plots] data={args.data}")
    print(f"[generate_shap_plots] output={args.output}")
    # TODO: compute SHAP values and render summary plot


if __name__ == "__main__":
    main()
