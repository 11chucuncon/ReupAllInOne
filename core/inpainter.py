from __future__ import annotations

import logging
import shlex
import subprocess
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


class VideoInpainter:
    """Run ProPainter inpainting on a video using masks to remove subtitles and watermarks."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.config_path = config_path or str(self.project_root / "config" / "settings.yaml")
        self.settings = self._load_settings()
        self.inpaint_config = self.settings.get("inpaint", {})
        self.propainter_dir = self.project_root / "core" / "ProPainter"

    def _load_settings(self) -> dict:
        try:
            with open(self.config_path, "r", encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
        except FileNotFoundError as exc:
            logger.error("Settings file not found at %s", self.config_path)
            raise
        except yaml.YAMLError as exc:
            logger.error("Invalid YAML configuration: %s", exc)
            raise

    def _run_command(self, command: list[str]) -> None:
        logger.info("Running inpainter command: %s", " ".join(shlex.quote(part) for part in command))
        subprocess.run(command, check=True)

    def _blur_video(self, input_video_path: str, output_video_path: str) -> str:
        output_path = Path(output_video_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-y",
            "-i",
            input_video_path,
            "-vf",
            "boxblur=10:1",
            "-c:a",
            "copy",
            str(output_path),
        ]
        self._run_command(command)
        return str(output_path)

    def clean_video(
        self,
        input_video_path: str,
        output_video_path: str,
        mode: str = "propainter",
        mask_video_path: Optional[str] = None,
    ) -> str:
        output_path = Path(output_video_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if mode == "blur":
            return self._blur_video(input_video_path, str(output_path))

        if not self.propainter_dir.exists():
            raise FileNotFoundError("ProPainter repository not found in core/ProPainter")

        command = [
            "python",
            str(self.propainter_dir / "run_video.py"),
            "--input",
            input_video_path,
            "--output",
            str(output_path),
            "--model_path",
            str(self.propainter_dir / "weights" / "ProPainter.pth"),
        ]
        if mask_video_path:
            command.extend(["--mask", mask_video_path])

        self._run_command(command)
        return str(output_path)
