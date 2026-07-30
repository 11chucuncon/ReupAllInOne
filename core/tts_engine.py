from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

import edge_tts
import yaml

logger = logging.getLogger(__name__)


class TTSEngine:
    """Generate speech audio from text using edge-tts."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config_path = config_path or str(Path(__file__).resolve().parents[1] / "config" / "settings.yaml")
        self.settings = self._load_settings()
        self.tts_config = self.settings.get("tts", {})
        self.default_voice = self.tts_config.get("default_voice", "vi-VN-HoaiMyNeural")
        self.default_rate = self.tts_config.get("rate", "+0%")
        self.default_volume = self.tts_config.get("volume", "+0%")

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

    async def generate_speech(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
    ) -> str:
        """Generate an MP3 audio file from the provided text."""
        if not text or not text.strip():
            raise ValueError("Text input cannot be empty")

        output_file = Path(output_path)
        if output_file.suffix.lower() != ".mp3":
            output_file = output_file.with_suffix(".mp3")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        selected_voice = voice or self.default_voice
        selected_rate = rate or self.default_rate

        try:
            communicate = edge_tts.Communicate(text, voice=selected_voice, rate=selected_rate)
            await communicate.save(str(output_file))
            logger.info("Generated speech audio at %s", output_file)
            return str(output_file)
        except Exception as exc:
            logger.exception("Text-to-speech generation failed")
            raise RuntimeError(f"TTS failed: {exc}") from exc
