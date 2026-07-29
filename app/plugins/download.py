from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yt_dlp

from app.plugins.base import BaseStep


def _resolve_local_source(source: str | Path) -> Path | None:
    path = Path(source)
    if path.exists() and path.is_file():
        return path.resolve()

    if path.parent.exists():
        matches = sorted(path.parent.glob(f"{path.name}*"))
        for match in matches:
            if match.is_file():
                return match.resolve()

    return None


class DownloadStep(BaseStep):
    name = "Download Video"

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        output_dir = Path(context.get("output_dir", "downloads"))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "raw.mp4"
        source = context.get("video_path") or context.get("input_path") or context.get("url")

        if not source:
            context["video_path"] = str(output_path)
            context["download_status"] = "failed"
            context["download_error"] = "No input URL or local video path provided"
            return context

        resolved_source = _resolve_local_source(source)
        if resolved_source is not None:
            context["video_path"] = str(resolved_source)
            context["download_status"] = "success"
            context["download_source"] = "local_file"
            return context

        ydl_opts = {
            "format": "bestvideo+bestaudio/best",
            "outtmpl": str(output_path),
            "quiet": True,
            "noplaylist": True,
            "postprocessors": [
                {
                    "key": "FFmpegVideoConvertor",
                    "preferedformat": "mp4",
                }
            ],
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([source])
            context["video_path"] = str(output_path)
            context["download_status"] = "success"
            context["download_source"] = "url"
        except Exception as exc:
            context["video_path"] = str(output_path)
            context["download_status"] = "failed"
            context["download_error"] = str(exc)

        return context
