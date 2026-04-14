"""CLI entry point for staged curriculum training with Transformer policy."""
import argparse

from carl.training.staged import train_staged_transformer
from carl.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Staged curriculum training with Transformer policy.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument("--run-name", type=str, default="staged_transformer_run", help="Name for this run.")
    parser.add_argument(
        "--stages",
        type=int,
        nargs="+",
        default=None,
        help="Stage indices to run (default: all stages).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    args = parser.parse_args()

    config = load_config(args.config)
    config.setdefault("run_name", args.run_name)
    config.setdefault("seed", args.seed)
    if args.stages is not None:
        config["stage_indices"] = args.stages

    train_staged_transformer(config)


if __name__ == "__main__":
    main()
