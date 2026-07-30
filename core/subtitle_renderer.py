from __future__ import annotations

import logging
import shlex
import subprocess
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class SubtitleRenderer:
    """Render dual or target subtitles on video with styling and overlay support."""

    def __init__(self, config_path: str | None = None) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.config_path = config_path or str(self.project_root / "config" / "settings.yaml")
        self.settings = self._load_settings()
        self.render_config = self.settings.get("subtitle", {})

    def _load_settings(self) -> dict[str, Any]:
        try:
            with open(self.config_path, "r", encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
        except FileNotFoundError as exc:
            logger.error("Settings file not found at %s", self.config_path)
            raise
        except yaml.YAMLError as exc:
            logger.error("Invalid YAML configuration: %s", exc)
            raise

    def _hex_to_ass_color(self, color: str) -> str:
        if not color:
            return "&H00FFFFFF&"
        hex_value = color.lstrip("#")
        if len(hex_value) == 3:
            hex_value = "".join(ch * 2 for ch in hex_value)
        if len(hex_value) != 6:
            return "&H00FFFFFF&"
        red = hex_value[0:2]
        green = hex_value[2:4]
        blue = hex_value[4:6]
        return f"&H00{blue}{green}{red}&"

    def build_filter(self, srt_path: str, mode: str, style: dict[str, str]) -> str:
        subtitle_style = (
            f"Fontname={style['font']},"
            f"Fontsize={style['size']},"
            f"PrimaryColour={self._hex_to_ass_color(style['color'])},"
            f"OutlineColour={self._hex_to_ass_color(style['border'])},"
            f"BackColour={self._hex_to_ass_color(style['shadow'])},"
            f"BorderStyle=1,Outline=2,Shadow=2"
        )
        escaped_path = srt_path.replace("'", "\\'")
        return f"subtitles='{escaped_path}':force_style='{subtitle_style}'"

    def render_subtitles(self, input_video_path: str, output_video_path: str, srt_path: str, mode: str, style: dict[str, str]) -> str:
        output_path = Path(output_video_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        filter_spec = self.build_filter(srt_path, mode, style)
        command = [
            "ffmpeg",
            "-y",
            "-i",
            input_video_path,
            "-vf",
            filter_spec,
            "-c:a",
            "copy",
            str(output_path),
        ]
        logger.info("Rendering subtitles with ffmpeg: %s", " ".join(shlex.quote(part) for part in command))
        subprocess.run(command, check=True)
        return str(output_path)
