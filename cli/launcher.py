"""Main launcher script that routes to training, evaluation, or audit."""
import argparse
import subprocess
import sys


_MODE_SCRIPTS = {
    "train": "cli/train_standard.py",
    "eval": "cli/eval.py",
    "audit": "cli/audit_run.py",
}


def main():
    parser = argparse.ArgumentParser(description="CARL main launcher.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=list(_MODE_SCRIPTS.keys()),
        required=True,
        help="Operation mode.",
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config file.")
    # Capture any extra arguments to forward to the sub-script.
    args, extra = parser.parse_known_args()

    script = _MODE_SCRIPTS[args.mode]
    cmd = [sys.executable, script]
    if args.config:
        cmd += ["--config", args.config]
    cmd += extra

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
