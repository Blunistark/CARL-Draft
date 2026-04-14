"""Scan / probe neural network activations for a trained CARL agent."""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Probe layer activations of a trained agent.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--data", type=str, required=True, help="Path to observation data file.")
    parser.add_argument("--layer", type=str, default=None, help="Layer name to probe (default: all layers).")
    args = parser.parse_args()

    print(f"[scan_brain] checkpoint={args.checkpoint}")
    print(f"[scan_brain] data={args.data}")
    print(f"[scan_brain] layer={args.layer or 'ALL'}")
    # TODO: load model, register forward hooks, collect activations


if __name__ == "__main__":
    main()
