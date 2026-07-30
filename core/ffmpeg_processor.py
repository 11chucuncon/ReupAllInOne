from __future__ import annotations

import logging
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

    def render_reup_video(
        self,
        video_path: str,
        new_audio_path: str,
        output_path: str,
        srt_path: Optional[str] = None,
    ) -> str:
        """Render a reup video with optional flipping, speed changes, audio replacement, and subtitles."""
        input_video = Path(video_path)
        input_audio = Path(new_audio_path)
        output_file = Path(output_path)

        if not input_video.exists():
            raise FileNotFoundError(f"Input video not found: {video_path}")
        if not input_audio.exists():
            raise FileNotFoundError(f"Input audio not found: {new_audio_path}")

        output_file.parent.mkdir(parents=True, exist_ok=True)

        command = ["ffmpeg", "-y", "-i", str(input_video), "-i", str(input_audio)]
        vf_parts = []

        if self.hflip:
            vf_parts.append("hflip")
        if srt_path:
            srt_file = Path(srt_path)
            if not srt_file.exists():
                raise FileNotFoundError(f"Subtitle file not found: {srt_path}")
            vf_parts.append(f"subtitles={srt_file.resolve().as_posix()}")
        if self.speed_factor != 1.0:
            vf_parts.append(f"setpts=PTS/{self.speed_factor}")

        if vf_parts:
            command.extend(["-vf", ",".join(vf_parts)])

        if self.speed_factor != 1.0:
            command.extend(["-af", f"atempo={self.speed_factor}"])

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

        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            logger.info("Rendered reup video at %s", output_file)
            return str(output_file)
        except subprocess.CalledProcessError as exc:
            logger.exception("FFmpeg rendering failed with stderr: %s", exc.stderr)
            raise RuntimeError(f"FFmpeg rendering failed: {exc.stderr}") from exc
