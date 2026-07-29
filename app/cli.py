from __future__ import annotations

import argparse

from app.core.pipeline import build_pipeline_plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a video processing pipeline plan")
    parser.add_argument("source_url", help="URL of the source video")
    args = parser.parse_args()

    plan = build_pipeline_plan(args.source_url)
    print(plan)


if __name__ == "__main__":
    main()
