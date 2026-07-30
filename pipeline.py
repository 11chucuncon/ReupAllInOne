from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Union

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
        self.project_root = Path(__file__).resolve().parent
        self.config_path = config_path or str(self.project_root / "config" / "settings.yaml")
        self.settings = self._load_settings()
        self.app_config = self.settings.get("app", {})

        self.downloader = VideoDownloader(config_path=self.config_path)
        self.transcriber = WhisperTranscriber(config_path=self.config_path)
        self.rewriter = LLMRewriter(config_path=self.config_path)
        self.tts_engine = TTSEngine(config_path=self.config_path)
        self.ffmpeg_processor = FFmpegProcessor(config_path=self.config_path)

        self.temp_dir = self._resolve_project_path(self.app_config.get("temp_dir", "temp"), default="temp")
        self.output_dir = self._resolve_project_path(self.app_config.get("output_dir", "outputs"), default="outputs")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_project_path(self, value: Optional[str], default: str) -> Path:
        """Resolve a path relative to the project root into an absolute path."""
        path = Path(value or default)
        if not path.is_absolute():
            path = self.project_root / path
        return path.expanduser().resolve()

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

    def _resolve_input_file(self, input_source: Union[str, Sequence[str], None]) -> str:
        """Resolve the input source to a local media file path."""
        if isinstance(input_source, (list, tuple)):
            for candidate in input_source:
                if isinstance(candidate, str) and candidate.strip():
                    input_source = candidate
                    break
            else:
                raise ValueError("No input source was provided")

        if not input_source or not isinstance(input_source, str):
            raise ValueError("Input source must be a non-empty string")

        source_value = input_source.strip()
        if source_value.startswith(("http://", "https://")):
            logger.info("[INFO] Step 1/5: Downloading video from URL...")
            downloaded_path = self.downloader.download(source_value, output_dir=str(self.temp_dir))
            logger.info("[INFO] Download completed: %s", downloaded_path)
            return downloaded_path

        local_path = Path(source_value)
        if not local_path.exists():
            raise FileNotFoundError(f"Input file not found: {source_value}")

        if local_path.is_file():
            logger.info("[INFO] Step 1/5: Using local file as input")
            return str(local_path.resolve())

        raise ValueError("Input source must be a valid file path or URL")

    def _format_srt_timestamp(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format."""
        total_milliseconds = int(seconds * 1000)
        hours, remainder = divmod(total_milliseconds, 3600 * 1000)
        minutes, remainder = divmod(remainder, 60 * 1000)
        seconds_part, milliseconds = divmod(remainder, 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds_part:02d},{milliseconds:03d}"

    def _write_srt_file(self, segments: Sequence[dict], output_path: Path) -> Path:
        """Write transcription segments to a .srt file in the temp directory."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []
        for index, segment in enumerate(segments, start=1):
            text = str(segment.get("text", "")).strip()
            if not text:
                continue
            start_time = self._format_srt_timestamp(float(segment.get("start", 0.0)))
            end_time = self._format_srt_timestamp(float(segment.get("end", segment.get("start", 0.0))))
            lines.extend([
                str(index),
                f"{start_time} --> {end_time}",
                text,
                "",
            ])

        if not lines:
            lines = [
                "1",
                "00:00:00,000 --> 00:00:01,000",
                "",
                "",
            ]

        with output_file.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(lines).strip() + "\n")
        logger.info("[INFO] Wrote subtitle file to %s", output_file)
        return output_file

    def _clean_temp_files(self) -> None:
        """Remove temporary files that are no longer needed after rendering."""
        try:
            for file_path in self.temp_dir.glob("*"):
                if file_path.is_file() and file_path.name not in {".keep"}:
                    file_path.unlink(missing_ok=True)
            logger.info("[INFO] Cleanup completed for temp directory")
        except Exception as exc:
            logger.warning("Cleanup failed: %s", exc)

    def process_video(
        self,
        input_source: Union[str, Sequence[str], None],
        auto_rewrite: bool = True,
        target_language: Optional[str] = "vi",
        custom_voice: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        tts_engine_mode: Optional[str] = "Edge-TTS Free (Tốc độ cao)",
        reference_audio_path: Optional[str] = None,
        voice_preset: Optional[str] = None,
        subtitle_font: Optional[str] = "Arial",
        subtitle_size: int = 32,
        subtitle_color: str = "#FFFFFF",
        subtitle_outline_color: str = "#000000",
        subtitle_position: str = "bottom",
        output_mode: str = "Keep original",
        speed_factor: float = 1.05,
        hflip: bool = True,
        background_audio_path: Optional[str] = None,
    ) -> str:
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
                logger.info("[INFO] Step 3/5: Rewriting script with OpenRouter AI...")
                self.rewriter.set_api_key(openrouter_api_key)
                rewritten_text = self.rewriter.rewrite(full_text, target_lang=target_language or "vi")
            else:
                logger.info("[INFO] Step 3/5: Using original transcript without rewrite")
                rewritten_text = full_text

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            srt_path = self.temp_dir / "sub.srt"
            self._write_srt_file(transcription.get("segments", []), srt_path)

            audio_output_path = self.temp_dir / "new_voice.mp3"
            logger.info("[INFO] Step 4/5: Generating new speech audio at %s...", audio_output_path)
            if str(tts_engine_mode or "").lower().startswith("local"):
                asyncio.run(
                    self.tts_engine.clone_speech(
                        rewritten_text,
                        str(audio_output_path),
                        reference_audio_path=reference_audio_path,
                        target_language=target_language,
                        voice=custom_voice,
                        voice_preset=voice_preset,
                    )
                )
            else:
                asyncio.run(
                    self.tts_engine.generate_speech(
                        rewritten_text,
                        str(audio_output_path),
                        voice=custom_voice,
                        engine_mode="edge",
                    )
                )

            output_video_path = self.output_dir / f"reup_{timestamp}.mp4"
            logger.info("[INFO] Step 5/5: Rendering final video...")
            rendered_path = self.ffmpeg_processor.render_reup_video(
                video_path=video_path,
                new_audio_path=str(audio_output_path),
                output_path=str(output_video_path),
                srt_path=str(srt_path),
                subtitle_text=rewritten_text,
                subtitle_font=subtitle_font or "Arial",
                subtitle_size=int(subtitle_size or 32),
                subtitle_color=subtitle_color or "#FFFFFF",
                subtitle_outline_color=subtitle_outline_color or "#000000",
                subtitle_position=subtitle_position or "bottom",
                output_mode=output_mode or "Keep original",
                speed_factor=float(speed_factor or 1.05),
                hflip=bool(hflip),
                background_audio_path=background_audio_path,
                target_language=target_language,
            )

            logger.info("[INFO] Step 6/5: Cleaning temporary files...")
            self._clean_temp_files()

            absolute_output = str(Path(rendered_path).resolve())
            logger.info("[INFO] Pipeline completed successfully: %s", absolute_output)
            return absolute_output

        except Exception as exc:
            logger.exception("[ERROR] Pipeline failed at runtime")
            raise RuntimeError(f"Pipeline failed: {exc}") from exc
