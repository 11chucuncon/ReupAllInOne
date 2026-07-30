from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

from core.inpainter import VideoInpainter
from core.ocr_processor import OCRProcessor
from core.subtitle_renderer import SubtitleRenderer
from core.translation import TranslationEngine
from core.tts_engine import TTSEngine
from core.upscaler import VideoUpscaler

logger = logging.getLogger(__name__)


class ReupPipeline:
    """Professional Auto-Reup pipeline for video OCR, inpainting, translation, TTS, and rendering."""

    def __init__(self, config_path: str | None = None) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.config_path = config_path or str(self.project_root / "config" / "settings.yaml")
        self.settings = self._load_settings()
        self.ocr_processor = OCRProcessor(self.config_path)
        self.inpainter = VideoInpainter(self.config_path)
        self.translator = TranslationEngine(self.config_path)
        self.subtitle_renderer = SubtitleRenderer(self.config_path)
        self.tts_engine = TTSEngine(self.config_path)
        self.upscaler = VideoUpscaler(self.config_path)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="reup_pipeline_"))

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

    def process_video(
        self,
        source_video: str,
        voice_name: str,
        tts_mode: str,
        target_language: str,
        subtitle_mode: str,
        inpaint_mode: str,
        output_format: str,
        upscale_target: str,
        subtitle_style: dict[str, str],
    ) -> str:
        logger.info("Starting professional video processing pipeline")

        cleaned_video_path = self.temp_dir / "cleaned_video.mp4"
        ocr_data = self.ocr_processor.detect_text(source_video)
        if inpaint_mode != "off":
            if inpaint_mode == "subtitle_mask":
                cleaned_video_path = self.inpainter.clean_video(source_video, str(cleaned_video_path))
            else:
                cleaned_video_path = self.inpainter.clean_video(source_video, str(cleaned_video_path), mask_video_path=str(source_video))
        else:
            cleaned_video_path = Path(source_video)

        srt_path = self._write_srt_file(ocr_data["segments"], self.temp_dir / "detected_subtitles.srt")
        translated_text = self.translator.translate_text(ocr_data["full_text"], target_language=target_language)
        translation_srt_path = self._write_srt_file(
            [{"index": item["index"], "start_time": item["start_time"], "end_time": item["end_time"], "text": translated_text} for item in ocr_data["segments"]],
            self.temp_dir / "translated_subtitles.srt",
        )

        combined_srt = translation_srt_path if subtitle_mode == "target" else srt_path
        if subtitle_mode == "dual":
            combined_srt = self._write_srt_file(
                [
                    {"index": item["index"], "start_time": item["start_time"], "end_time": item["end_time"], "text": f"{item['text']} | {translated_text}"}
                    for item in ocr_data["segments"]
                ],
                self.temp_dir / "dual_subtitles.srt",
            )

        rendered_with_subtitles = self.subtitle_renderer.render_subtitles(
            str(cleaned_video_path),
            str(self.temp_dir / "rendered_with_subtitles.mp4"),
            str(combined_srt),
            subtitle_mode,
            subtitle_style,
        )

        audio_path = self.tts_engine.generate_speech(
            translated_text if tts_mode == "translated" else ocr_data["full_text"],
            voice_name,
            self.temp_dir / "tts_audio.mp3",
        )

        final_path = self._merge_audio(rendered_with_subtitles, audio_path, self.temp_dir / "final_output.mp4")

        if upscale_target != "none":
            upscaled_path = self.upscaler.upscale_video(final_path, target_resolution=upscale_target)
            final_path = upscaled_path

        output_path = self.project_root / "output" / f"reup_output_{subtitle_mode}_{upscale_target}.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(final_path, output_path)
        self._clean_temp_files()

        return str(output_path)

    def _write_srt_file(self, segments: list[dict[str, Any]], path: Path) -> Path:
        with open(path, "w", encoding="utf-8") as handle:
            for segment in segments:
                handle.write(f"{segment['index']}\n")
                handle.write(f"{self._format_srt_time(segment['start_time'])} --> {self._format_srt_time(segment['end_time'])}\n")
                handle.write(f"{segment['text']}\n\n")
        return path

    def _format_srt_time(self, value: float) -> str:
        hours = int(value // 3600)
        minutes = int((value % 3600) // 60)
        seconds = int(value % 60)
        milliseconds = int((value - int(value)) * 1000)
        return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

    def _merge_audio(self, video_path: str, audio_path: str, output_path: Path) -> str:
        command = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ]
        logger.info("Merging audio with ffmpeg: %s", " ".join(command))
        subprocess.run(command, check=True)
        return str(output_path)

    def _clean_temp_files(self) -> None:
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
