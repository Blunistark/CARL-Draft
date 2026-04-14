"""Generate control-surface time-series plots (elevator, aileron, rudder)."""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Plot control surface deflections over time.")
    parser.add_argument("--trajectory", type=str, required=True, help="Path to trajectory file (CSV/JSON).")
    parser.add_argument("--output", type=str, required=True, help="Output image file path.")
    args = parser.parse_args()

    print(f"[generate_control_plots] trajectory={args.trajectory}")
    print(f"[generate_control_plots] output={args.output}")
    # TODO: load trajectory, plot elevator/aileron/rudder columns, save figure


if __name__ == "__main__":
    main()
