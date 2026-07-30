from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """Transcribe audio/video media into text using faster-whisper."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config_path = config_path or str(Path(__file__).resolve().parents[1] / "config" / "settings.yaml")
        self.settings = self._load_settings()
        self.whisper_config = self.settings.get("whisper", {})

        model_size = self.whisper_config.get("model_size", "small")
        device = self.whisper_config.get("device", "cuda")
        compute_type = self.whisper_config.get("compute_type", "float16")

        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception as exc:
            logger.exception("Unable to initialize Whisper model")
            raise RuntimeError(f"Whisper initialization failed: {exc}") from exc

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

    def transcribe(self, media_path: str) -> dict:
        """Transcribe audio content and return full text plus segments."""
        media_file = Path(media_path)
        if not media_file.exists():
            raise FileNotFoundError(f"Media file not found: {media_path}")

        try:
            segments, info = self.model.transcribe(str(media_file))
            segment_list = []
            full_text_parts = []

            for segment in segments:
                text = segment.text.strip()
                if not text:
                    continue
                full_text_parts.append(text)
                segment_list.append({
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": text,
                })

            return {
                "full_text": " ".join(full_text_parts),
                "segments": segment_list,
                "language": info.language if hasattr(info, "language") else None,
            }
        except Exception as exc:
            logger.exception("Whisper transcription failed for %s", media_path)
            raise RuntimeError(f"Transcription failed: {exc}") from exc
