import argparse
import sys
from pathlib import Path

from app.plugins.runner import run_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the plugin-based video pipeline from a YAML config")
    parser.add_argument("config", nargs="?", default="config_pipeline_full.yaml", help="Path to YAML config")
    parser.add_argument("video_input", nargs="?", default=None, help="Optional local video path or remote URL to process")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        raise SystemExit(1)

    result = run_from_config(config_path, video_url=args.video_input)
    print(result)


if __name__ == "__main__":
    main()
