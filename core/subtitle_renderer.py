from __future__ import annotations

import logging
import os
import re
import shlex
import shutil
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

    def escape_ass_path(self, path: str) -> str:
        normalized = str(path).replace("\\", "/")
        normalized = normalized.replace(":", "\\:")
        return normalized.replace("'", "''")

    def _format_ass_timestamp(self, raw_value: str) -> str:
        cleaned = raw_value.strip().replace(",", ".")
        if cleaned.count(":") == 2:
            hours, minutes, seconds = cleaned.split(":")
            return f"{hours}:{minutes}:{seconds}"
        return cleaned

    def _parse_srt_blocks(self, srt_path: str) -> list[dict[str, str]]:
        content = Path(srt_path).read_text(encoding="utf-8")
        blocks = [block.strip() for block in content.split("\n\n") if block.strip()]
        parsed: list[dict[str, str]] = []
        timestamp_pattern = re.compile(r"(\d{2}:\d{2}:\d{2}[,.]\d{3}) --> (\d{2}:\d{2}:\d{2}[,.]\d{3})")
        for block in blocks:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if len(lines) < 3:
                continue
            match = timestamp_pattern.search(lines[1])
            if not match:
                continue
            parsed.append({
                "start": self._format_ass_timestamp(match.group(1)),
                "end": self._format_ass_timestamp(match.group(2)),
                "text": " ".join(lines[2:]),
            })
        return parsed

    def write_ass_file(self, srt_path: str, output_ass_path: str, style: dict[str, str]) -> Path:
        output_path = Path(output_ass_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subtitle_style = (
            f"Name=Default,"
            f"Fontname={style.get('font', 'DejaVu Sans')},"
            f"Fontsize={style.get('size', '32')},"
            f"PrimaryColour={self._hex_to_ass_color(style.get('color', '#FFFFFF'))},"
            f"OutlineColour={self._hex_to_ass_color(style.get('border', '#000000'))},"
            f"BackColour={self._hex_to_ass_color(style.get('shadow', '#000000'))},"
            f"Bold=0,Italic=0,Underline=0,StrikeOut=0,ScaleX=100,ScaleY=100,"
            f"Spacing=0,Angle=0,BorderStyle=1,Outline=2,Shadow=2,Alignment=2,MarginL=20,MarginR=20,MarginV=20"
        )
        style_fields = [
            f"Fontname={style.get('font', 'DejaVu Sans')}",
            f"Fontsize={style.get('size', '32')}",
            f"PrimaryColour={self._hex_to_ass_color(style.get('color', '#FFFFFF'))}",
            f"SecondaryColour=&H00000000&",
            f"OutlineColour={self._hex_to_ass_color(style.get('border', '#000000'))}",
            f"BackColour={self._hex_to_ass_color(style.get('shadow', '#000000'))}",
            "Bold=0",
            "Italic=0",
            "Underline=0",
            "StrikeOut=0",
            "ScaleX=100",
            "ScaleY=100",
            "Spacing=0",
            "Angle=0",
            "BorderStyle=1",
            "Outline=2",
            "Shadow=2",
            "Alignment=2",
            "MarginL=20",
            "MarginR=20",
            "MarginV=20",
            "Encoding=1",
        ]
        ass_lines = [
            "[Script Info]",
            "Title: Reup Subtitles",
            "ScriptType: v4.00+",
            "WrapStyle: 0",
            "ScaledBorderAndShadow: yes",
            "YCbCr Matrix: TV.601",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            f"Style: Default,{','.join(style_fields)}",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]

        for segment in self._parse_srt_blocks(srt_path):
            safe_text = segment["text"].replace("\n", " ").replace("|", "\\|")
            ass_lines.append(
                f"Dialogue: 0,{segment['start']},{segment['end']},Default,,0,0,0,,{safe_text}"
            )

        output_path.write_text("\n".join(ass_lines) + "\n", encoding="utf-8")
        return output_path

    def build_filter(self, ass_path: str, mode: str, style: dict[str, str]) -> str:
        escaped_path = self.escape_ass_path(ass_path)
        return f"ass='{escaped_path}'"

    def _resolve_media_input_path(self, input_video_path: str) -> Path:
        input_path = Path(input_video_path).expanduser().resolve()
        if input_path.exists() and input_path.is_dir():
            video_candidates = sorted(
                path
                for path in input_path.rglob("*")
                if path.is_file() and path.suffix.lower() in {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}
            )
            if video_candidates:
                return video_candidates[0]
            raise RuntimeError(f"Input media path is a directory and contains no video files: {input_path}")
        if not input_path.exists():
            raise FileNotFoundError(f"Input media file does not exist: {input_path}")
        return input_path

    def render_subtitles(self, input_video_path: str, output_video_path: str, srt_path: str, mode: str, style: dict[str, str]) -> str:
        output_path = Path(output_video_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and output_path.is_dir():
            shutil.rmtree(output_path)
        elif output_path.exists():
            os.remove(output_path)
        ass_path = output_path.with_suffix(".ass")
        self.write_ass_file(srt_path, str(ass_path), style)
        filter_spec = self.build_filter(str(ass_path), mode, style)
        input_path = self._resolve_media_input_path(input_video_path)
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-vf",
            filter_spec,
            "-c:a",
            "copy",
            str(output_path),
        ]
        logger.info("Rendering subtitles with ffmpeg: %s", " ".join(shlex.quote(part) for part in command))
        try:
            completed = subprocess.run(command, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            stderr_output = exc.stderr or ""
            logger.error("FFmpeg subtitle rendering failed with stderr:\n%s", stderr_output)
            raise RuntimeError(f"FFmpeg subtitle rendering failed: {stderr_output}") from exc
        if completed.stderr:
            logger.info("FFmpeg subtitle rendering stderr: %s", completed.stderr)
        return str(output_path)
