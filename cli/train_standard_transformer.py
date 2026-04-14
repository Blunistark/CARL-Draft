"""CLI entry point for training a standard TD3 agent with Transformer policy."""
import argparse

from carl.training.standard import train_standard_transformer
from carl.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Train a standard TD3+Transformer agent.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--run-name", type=str, default="standard_transformer_run", help="Name for this run.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device (cpu/cuda).")
    args = parser.parse_args()

    config = load_config(args.config)
    config.setdefault("run_name", args.run_name)
    config.setdefault("seed", args.seed)
    config.setdefault("device", args.device)

    train_standard_transformer(config)


if __name__ == "__main__":
    main()
