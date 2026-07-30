from __future__ import annotations

import logging
import shlex
import subprocess
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


class VideoUpscaler:
    """Upscale video quality using Real-ESRGAN and preserve original audio."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.config_path = config_path or str(self.project_root / "config" / "settings.yaml")
        self.settings = self._load_settings()
        self.upscale_config = self.settings.get("upscale", {})
        self.model_path = self._resolve_model_path(self.upscale_config.get("model_path", "weights/realesr-animevideov3.pth"))
        self.device = self.upscale_config.get("device", "cuda")
        self.temp_dir = self.project_root / self.upscale_config.get("temp_dir", "temp/upscale")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _load_settings(self) -> dict:
        """Load project settings from config/settings.yaml."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
        except FileNotFoundError as exc:
            logger.error("Settings file not found at %s", self.config_path)
            raise FileNotFoundError(f"Missing configuration file: {self.config_path}") from exc
        except yaml.YAMLError as exc:
            logger.error("Failed to parse settings YAML: %s", exc)
            raise RuntimeError("Invalid YAML configuration") from exc

    def _resolve_model_path(self, model_path: str) -> Path:
        path = Path(model_path)
        if not path.is_absolute():
            path = self.project_root / path
        return path.expanduser().resolve()

    def _run_command(self, command: list[str]) -> None:
        command_str = " ".join(shlex.quote(part) for part in command)
        logger.info("Running upscale command: %s", command_str)
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("Upscale command failed: %s", result.stderr)
            raise RuntimeError(f"Upscale process failed: {result.stderr}")

    def _extract_audio(self, source_video: Path, audio_path: Path) -> None:
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(source_video),
            "-vn",
            "-acodec",
            "copy",
            str(audio_path),
        ]
        self._run_command(command)

    def _attach_audio(self, source_video: Path, audio_path: Path, output_video: Path) -> None:
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(source_video),
            "-i",
            str(audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-movflags",
            "+faststart",
            str(output_video),
        ]
        self._run_command(command)

    def _upscale_video(self, input_video: Path, output_video: Path, scale: int) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Real-ESRGAN model weights not found at {self.model_path}."
                " Please download the model weights to the configured path."
            )

        command = [
            "python",
            "-m",
            "realesrgan",
            "--input",
            str(input_video),
            "--output",
            str(output_video),
            "--scale",
            str(scale),
            "--model_path",
            str(self.model_path),
        ]
        self._run_command(command)

    def upscale(self, input_video_path: str, output_video_path: str, scale: int = 2) -> str:
        """Upscale a video file and preserve original audio."""
        if scale not in {2, 4}:
            raise ValueError("Scale must be either 2 or 4")

        input_video = Path(input_video_path).expanduser().resolve()
        output_video = Path(output_video_path).expanduser().resolve()
        output_video.parent.mkdir(parents=True, exist_ok=True)

        if not input_video.exists():
            raise FileNotFoundError(f"Input video not found: {input_video}")

        temp_audio = self.temp_dir / "upscale_audio.aac"
        temp_video = self.temp_dir / f"upscale_temp_{scale}x.mp4"

        logger.info("Extracting audio from source video %s", input_video)
        self._extract_audio(input_video, temp_audio)

        logger.info("Upscaling video %s with scale %sx", input_video, scale)
        self._upscale_video(input_video, temp_video, scale)

        logger.info("Reattaching audio to upscaled video %s", temp_video)
        self._attach_audio(temp_video, temp_audio, output_video)
        logger.info("Upscaled video saved to %s", output_video)

        return str(output_video)
