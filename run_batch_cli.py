import argparse
import sys
from pathlib import Path

from app.plugins.batch_runner import run_batch_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the plugin pipeline for multiple videos in batch")
    parser.add_argument("config", nargs="?", default="config_pipeline_full.yaml", help="Path to YAML config")
    parser.add_argument("videos", nargs="+", help="One or more local files or remote URLs")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        raise SystemExit(1)

    results = run_batch_from_config(config_path, args.videos)
    for result in results:
        print(result.get("rendered_path"), result.get("render_status"))


if __name__ == "__main__":
    main()
