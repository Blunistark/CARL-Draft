"""CLI entry point for evaluating a trained agent."""
import argparse

from carl.evaluation.eval_drl import evaluate_agent
from carl.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained DRL agent.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to agent checkpoint directory.")
    parser.add_argument("--env-config", type=str, required=True, help="Path to environment YAML config.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes.")
    parser.add_argument("--render", action="store_true", help="Render the environment during evaluation.")
    parser.add_argument("--output", type=str, default=None, help="Path to save evaluation results (JSON).")
    args = parser.parse_args()

    # NOTE: Loading the agent and environment from config paths is project-specific.
    # The example below shows the intended usage pattern; replace the stubs with
    # real agent / env construction once those helpers exist.
    env_config = load_config(args.env_config)
    print(f"[eval] checkpoint={args.checkpoint} episodes={args.episodes}")
    print(f"[eval] env_config={env_config}")
    # agent = <load agent from args.checkpoint>
    # env   = <build env from env_config>
    # metrics = evaluate_agent(agent=agent, env=env, n_episodes=args.episodes, render=args.render)
    # if args.output:
    #     import json
    #     with open(args.output, "w") as f:
    #         json.dump(metrics, f, indent=2)
    raise NotImplementedError(
        "eval.py requires agent and environment loaders — implement once those helpers exist."
    )


if __name__ == "__main__":
    main()
