"""Generate SHAP beeswarm plots for a trained CARL agent."""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Generate SHAP beeswarm plots.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--data", type=str, required=True, help="Path to observation data file.")
    parser.add_argument("--output", type=str, required=True, help="Output image file path.")
    args = parser.parse_args()

    print(f"[generate_shap_beeswarm] checkpoint={args.checkpoint}")
    print(f"[generate_shap_beeswarm] data={args.data}")
    print(f"[generate_shap_beeswarm] output={args.output}")
    # TODO: compute SHAP values and render beeswarm plot


if __name__ == "__main__":
    main()
