"""SHAP / gradient-based model explanation for a trained CARL agent.

Stub implementation — extend with shap or captum backend.
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Generate model explanations using SHAP or gradients.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--data", type=str, required=True, help="Path to observation data file.")
    parser.add_argument("--output", type=str, required=True, help="Output directory for explanation plots.")
    args = parser.parse_args()

    print(f"[explain_model] checkpoint={args.checkpoint}")
    print(f"[explain_model] data={args.data}")
    print(f"[explain_model] output={args.output}")
    # TODO: load model, compute SHAP values, save plots


if __name__ == "__main__":
    main()
