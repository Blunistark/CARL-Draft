"""CLI entry point for fine-tuning an agent for action smoothness."""
import argparse

from carl.training.tuning import fine_tune_smoothness
from carl.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a trained agent for action smoothness.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to agent checkpoint directory.")
    parser.add_argument("--config", type=str, required=True, help="Path to fine-tuning YAML config file.")
    parser.add_argument("--run-name", type=str, default="finetune_smooth", help="Name for this run.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device (cpu/cuda).")
    args = parser.parse_args()

    config = load_config(args.config)
    config.setdefault("run_name", args.run_name)
    config.setdefault("seed", args.seed)
    config.setdefault("device", args.device)

    fine_tune_smoothness(agent_path=args.checkpoint, config=config)


if __name__ == "__main__":
    main()
