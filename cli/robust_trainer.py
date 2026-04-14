"""Robust training wrapper with retry/recovery logic."""
import argparse
import time

from carl.training.standard import train_standard
from carl.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Robust training with automatic retry on failure.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--run-name", type=str, default="robust_run", help="Name for this run.")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retry attempts.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device (cpu/cuda).")
    args = parser.parse_args()

    for attempt in range(1, args.max_retries + 1):
        try:
            print(f"Training attempt {attempt}/{args.max_retries} ...")
            config = load_config(args.config)
            config.setdefault("run_name", args.run_name)
            config.setdefault("seed", args.seed)
            config.setdefault("device", args.device)
            train_standard(config)
            print("Training completed successfully.")
            return
        except Exception as exc:  # noqa: BLE001
            print(f"Attempt {attempt} failed: {exc}")
            if attempt < args.max_retries:
                wait = 2 ** attempt
                print(f"Retrying in {wait}s ...")
                time.sleep(wait)

    raise RuntimeError(f"Training failed after {args.max_retries} attempts.")


if __name__ == "__main__":
    main()
