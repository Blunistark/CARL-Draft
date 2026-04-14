"""CLI script that audits a completed training run and generates a report."""
import argparse
import json
import os
import sys


def _check_run_dir(run_dir: str) -> list[str]:
    """Return a list of issues found in the run directory."""
    issues = []
    if not os.path.isdir(run_dir):
        issues.append(f"Run directory does not exist: {run_dir}")
        return issues

    expected = ["config.yaml", "checkpoints"]
    for name in expected:
        if not os.path.exists(os.path.join(run_dir, name)):
            issues.append(f"Missing expected path: {name}")

    return issues


def main():
    parser = argparse.ArgumentParser(description="Audit a training run and generate a report.")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to the run directory.")
    parser.add_argument("--output", type=str, default=None, help="Path to write the audit report (JSON).")
    args = parser.parse_args()

    issues = _check_run_dir(args.run_dir)

    report = {
        "run_dir": args.run_dir,
        "issues": issues,
        "status": "PASS" if not issues else "FAIL",
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Audit report written to {args.output}")
    else:
        print(json.dumps(report, indent=2))

    if issues:
        sys.exit(1)


if __name__ == "__main__":
    main()
