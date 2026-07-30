from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from core.downloader import VideoDownloader
from core.ffmpeg_processor import FFmpegProcessor
from core.rewriter import LLMRewriter
from core.transcriber import WhisperTranscriber
from core.tts_engine import TTSEngine

logger = logging.getLogger(__name__)


class ReupPipeline:
    """Coordinate the full auto-reup video workflow."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config_path = config_path or str(Path(__file__).resolve().parent / "config" / "settings.yaml")
        self.settings = self._load_settings()
        self.app_config = self.settings.get("app", {})

        self.downloader = VideoDownloader(config_path=self.config_path)
        self.transcriber = WhisperTranscriber(config_path=self.config_path)
        self.rewriter = LLMRewriter(config_path=self.config_path)
        self.tts_engine = TTSEngine(config_path=self.config_path)
        self.ffmpeg_processor = FFmpegProcessor(config_path=self.config_path)

        self.temp_dir = Path(self.app_config.get("temp_dir", "temp"))
        self.output_dir = Path(self.app_config.get("output_dir", "outputs"))
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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

    def _resolve_input_file(self, input_source: str) -> str:
        """Resolve the input source to a local media file path."""
        if not input_source or not isinstance(input_source, str):
            raise ValueError("Input source must be a non-empty string")

        if input_source.startswith(("http://", "https://")):
            logger.info("[INFO] Step 1/5: Downloading video from URL...")
            downloaded_path = self.downloader.download(input_source, output_dir=str(self.temp_dir))
            logger.info("[INFO] Download completed: %s", downloaded_path)
            return downloaded_path

        local_path = Path(input_source)
        if not local_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_source}")

        if local_path.is_file():
            logger.info("[INFO] Step 1/5: Using local file as input")
            return str(local_path.resolve())

        raise ValueError("Input source must be a valid file path or URL")

    def _clean_temp_files(self) -> None:
        """Remove temporary files that are no longer needed after rendering."""
        try:
            for file_path in self.temp_dir.glob("*"):
                if file_path.is_file() and file_path.name not in {".keep"}:
                    file_path.unlink(missing_ok=True)
            logger.info("[INFO] Cleanup completed for temp directory")
        except Exception as exc:
            logger.warning("Cleanup failed: %s", exc)

    def process_video(self, input_source: str, auto_rewrite: bool = True, custom_voice: Optional[str] = None) -> str:
        """Run the full video reup pipeline and return the output video path."""
        try:
            logger.info("[INFO] Starting pipeline for input: %s", input_source)

            video_path = self._resolve_input_file(input_source)

            logger.info("[INFO] Step 2/5: Transcribing video content...")
            transcription = self.transcriber.transcribe(video_path)
            full_text = transcription.get("full_text", "")
            if not full_text.strip():
                raise ValueError("Transcription produced empty text")

            if auto_rewrite:
                logger.info("[INFO] Step 3/5: Rewriting script with Gemini AI...")
                rewritten_text = self.rewriter.rewrite(full_text)
            else:
                logger.info("[INFO] Step 3/5: Using original transcript without rewrite")
                rewritten_text = full_text

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_output_path = self.temp_dir / f"new_voice_{timestamp}.mp3"
            logger.info("[INFO] Step 4/5: Generating new speech audio...")
            asyncio.run(
                self.tts_engine.generate_speech(
                    rewritten_text,
                    str(audio_output_path),
                    voice=custom_voice,
                )
            )

            output_video_path = self.output_dir / f"reup_{timestamp}.mp4"
            logger.info("[INFO] Step 5/5: Rendering final video...")
            rendered_path = self.ffmpeg_processor.render_reup_video(
                video_path=video_path,
                new_audio_path=str(audio_output_path),
                output_path=str(output_video_path),
            )

            logger.info("[INFO] Step 6/5: Cleaning temporary files...")
            self._clean_temp_files()

            absolute_output = str(Path(rendered_path).resolve())
            logger.info("[INFO] Pipeline completed successfully: %s", absolute_output)
            return absolute_output

        except Exception as exc:
            logger.exception("[ERROR] Pipeline failed at runtime")
            raise RuntimeError(f"Pipeline failed: {exc}") from exc
