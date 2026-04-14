"""Generate the full suite of audit plots for a completed training run."""
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Generate full audit plot suite for a training run.")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to the training run directory.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write output plots.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[generate_full_audit_plots] run_dir={args.run_dir}")
    print(f"[generate_full_audit_plots] output_dir={args.output_dir}")
    # TODO: call generate_3d_plots, generate_control_plots, compare_smoothness, etc.


if __name__ == "__main__":
    main()
