from __future__ import annotations

import logging
import shlex
import subprocess
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


class FFmpegProcessor:
    """Render final reup videos using FFmpeg CLI."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config_path = config_path or str(Path(__file__).resolve().parents[1] / "config" / "settings.yaml")
        self.settings = self._load_settings()
        self.ffmpeg_config = self.settings.get("ffmpeg", {})
        self.hflip = self.ffmpeg_config.get("hflip", True)
        self.speed_factor = float(self.ffmpeg_config.get("speed_factor", 1.05))
        self.use_nvenc = self.ffmpeg_config.get("use_nvenc", True)

    def _load_settings(self) -> dict:
        """Load configuration from config/settings.yaml."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
        except FileNotFoundError as exc:
            logger.error("Settings file not found at %s", self.config_path)
            raise FileNotFoundError(f"Missing configuration file: {self.config_path}") from exc
        except yaml.YAMLError as exc:
            logger.error("Failed to parse settings YAML: %s", exc)
            raise RuntimeError("Invalid YAML configuration") from exc

    def _hex_to_ass_color(self, color: str) -> str:
        """Convert a hex string to an ASS-compatible color."""
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

    def _create_ass_subtitles(
        self,
        output_path: Path,
        subtitle_text: str,
        subtitle_font: str,
        subtitle_size: int,
        subtitle_color: str,
        subtitle_outline_color: str,
        subtitle_position: str,
    ) -> Path:
        """Create a simple ASS subtitle file for FFmpeg styling."""
        subtitle_file = output_path.with_suffix(".ass")
        alignment = "2" if subtitle_position.lower() == "bottom" else "5"
        margin_v = "40" if subtitle_position.lower() == "bottom" else "0"
        content = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Alignment, BorderStyle, Outline, Shadow, MarginL, MarginR, MarginV
Style: Default,{subtitle_font},{subtitle_size},{self._hex_to_ass_color(subtitle_color)},{self._hex_to_ass_color(subtitle_color)},{self._hex_to_ass_color(subtitle_outline_color)},{self._hex_to_ass_color(subtitle_outline_color)},1,0,{alignment},1,2,0,10,10,{margin_v}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:10:00.00,Default,,10,10,{margin_v},,{{\\an{alignment}}}{subtitle_text}
"""
        subtitle_file.write_text(content, encoding="utf-8")
        return subtitle_file

    def render_reup_video(
        self,
        video_path: str,
        new_audio_path: str,
        output_path: str,
        srt_path: Optional[str] = None,
        subtitle_text: Optional[str] = None,
        subtitle_font: str = "Arial",
        subtitle_size: int = 32,
        subtitle_color: str = "#FFFFFF",
        subtitle_outline_color: str = "#000000",
        subtitle_position: str = "bottom",
        output_mode: str = "Keep original",
        speed_factor: Optional[float] = None,
        hflip: bool = True,
        background_audio_path: Optional[str] = None,
    ) -> str:
        """Render a reup video with optional flipping, speed changes, audio replacement, subtitles, and ratio adjustments."""
        input_video = Path(video_path).expanduser().resolve()
        input_audio = Path(new_audio_path).expanduser().resolve()
        output_file = Path(output_path).expanduser().resolve()

        if not input_video.exists():
            raise FileNotFoundError(f"Input video not found: {video_path}")
        if not input_audio.exists():
            raise FileNotFoundError(f"Input audio not found: {new_audio_path}")

        output_file.parent.mkdir(parents=True, exist_ok=True)

        command = ["ffmpeg", "-y", "-i", str(input_video), "-i", str(input_audio)]
        vf_parts = []

        if hflip:
            vf_parts.append("hflip")
        if output_mode == "Vertical 9:16 (Shorts/TikTok)":
            vf_parts.append("scale=1080:-2:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2")
        elif output_mode == "Horizontal 16:9 (YouTube)":
            vf_parts.append("scale=1920:-2:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2")

        subtitle_file = None
        if srt_path:
            srt_file = Path(srt_path).expanduser().resolve()
            if not srt_file.exists():
                raise FileNotFoundError(f"Subtitle file not found: {srt_path}")
            subtitle_file = srt_file
        elif subtitle_text and subtitle_text.strip():
            subtitle_file = self._create_ass_subtitles(
                output_path=output_file,
                subtitle_text=subtitle_text.strip(),
                subtitle_font=subtitle_font,
                subtitle_size=subtitle_size,
                subtitle_color=subtitle_color,
                subtitle_outline_color=subtitle_outline_color,
                subtitle_position=subtitle_position,
            )

        if subtitle_file:
            subtitle_filter = (
                f"subtitles={subtitle_file.as_posix()}:"
                f"force_style='Fontname={subtitle_font},Fontsize={subtitle_size},"
                f"PrimaryColour={self._hex_to_ass_color(subtitle_color)},"
                f"OutlineColour={self._hex_to_ass_color(subtitle_outline_color)}'"
            )
            vf_parts.append(subtitle_filter)

        speed_value = self.speed_factor if speed_factor is None else speed_factor
        if speed_value != 1.0:
            vf_parts.append(f"setpts=PTS/{speed_value}")

        if vf_parts:
            command.extend(["-vf", ",".join(vf_parts)])

        if background_audio_path:
            background_audio = Path(background_audio_path).expanduser().resolve()
            if not background_audio.exists():
                raise FileNotFoundError(f"Background audio file not found: {background_audio_path}")
            command.extend(["-i", str(background_audio)])
            command.extend(["-filter_complex", "[1:a][2:a]amix=inputs=2:duration=longest[a]", "-map", "0:v", "-map", "[a]"])
        else:
            command.extend(["-map", "0:v", "-map", "1:a"])

        if speed_value != 1.0 and not background_audio_path:
            command.extend(["-af", f"atempo={speed_value}"])

        codec = "h264_nvenc" if self.use_nvenc else "libx264"
        command.extend([
            "-c:v",
            codec,
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output_file),
        ])

        ffmpeg_command = " ".join(shlex.quote(part) for part in command)
        logger.info("Running FFmpeg command: %s", ffmpeg_command)
        print(f"[FFmpeg] {ffmpeg_command}")

        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            logger.info("Rendered reup video at %s", output_file)
            return str(output_file)
        except subprocess.CalledProcessError as exc:
            logger.exception("FFmpeg rendering failed with stderr: %s", exc.stderr)
            raise RuntimeError(f"FFmpeg rendering failed: {exc.stderr}") from exc
