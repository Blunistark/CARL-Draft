"""Animate feature importance scores over a trajectory.

Stub implementation — extend with a real importance-scoring backend.
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Animate feature importance over a trajectory.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--trajectory", type=str, required=True, help="Path to trajectory file.")
    parser.add_argument("--output", type=str, required=True, help="Output animation file path.")
    args = parser.parse_args()

    print(f"[animate_importance] checkpoint={args.checkpoint}")
    print(f"[animate_importance] trajectory={args.trajectory}")
    print(f"[animate_importance] output={args.output}")
    # TODO: implement importance scoring and animation


if __name__ == "__main__":
    main()
