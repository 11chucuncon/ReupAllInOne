from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml
import yt_dlp

logger = logging.getLogger(__name__)


class VideoDownloader:
    """Download videos from a URL using yt-dlp."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config_path = config_path or str(Path(__file__).resolve().parents[1] / "config" / "settings.yaml")
        self.settings = self._load_settings()
        self.app_config = self.settings.get("app", {})

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

    def download(self, url: str, output_dir: Optional[str] = None) -> str:
        """Download the highest available MP4 video and return its file path."""
        if not url or not isinstance(url, str):
            raise ValueError("A valid video URL is required")

        project_root = Path(__file__).resolve().parents[1]
        target_dir = Path(output_dir or self.app_config.get("output_dir", "outputs"))
        if not target_dir.is_absolute():
            target_dir = project_root / target_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": str(target_dir / "%(title)s.%(ext)s"),
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(url, download=True)
        except Exception as exc:
            logger.exception("Failed to download video from URL: %s", url)
            raise RuntimeError(f"Video download failed: {exc}") from exc

        try:
            downloaded_files = sorted(target_dir.rglob("*.mp4"), key=lambda path: path.stat().st_mtime, reverse=True)
            if not downloaded_files:
                raise FileNotFoundError("No MP4 file was generated after download")
            downloaded_file = downloaded_files[0]
            logger.info("Downloaded video saved to %s", downloaded_file)
            return str(downloaded_file)
        except FileNotFoundError as exc:
            logger.error("No download output found in %s", target_dir)
            raise RuntimeError("Download completed but no output file was found") from exc
