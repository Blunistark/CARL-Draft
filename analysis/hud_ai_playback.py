"""HUD-style AI playback visualizer for recorded trajectories."""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Play back a trajectory with a HUD-style overlay.")
    parser.add_argument("--trajectory", type=str, required=True, help="Path to trajectory file.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument("--output", type=str, default=None, help="Path to save output video (optional).")
    args = parser.parse_args()

    print(f"[hud_ai_playback] trajectory={args.trajectory} speed={args.speed}")
    if args.output:
        print(f"[hud_ai_playback] saving to {args.output}")
    # TODO: implement HUD rendering and animation


if __name__ == "__main__":
    main()
