from __future__ import annotations

import asyncio
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Union

import yaml

from config import (
    AUDIO_OUTPUT_PATH,
    CLEANED_DIR,
    CLEANED_VIDEO_PATH,
    FINAL_VIDEO_PATH,
    INPUT_DIR,
    OUTPUT_DIR,
    SUBTITLE_ASS_PATH,
    SUBTITLE_SRT_PATH,
    TEMP_DIR,
    find_file_anywhere,
    initialize_workspace,
    promote_file_to_destination,
    resolve_workspace_media_file,
    safe_file_path,
)

from core.downloader import VideoDownloader
from core.ffmpeg_processor import FFmpegProcessor
from core.inpainter import VideoInpainter
from core.rewriter import LLMRewriter
from core.transcriber import WhisperTranscriber
from core.tts_engine import TTSEngine
from core.translator import TranslationEngine
from core.upscaler import VideoUpscaler
from core.subtitle_renderer import SubtitleRenderer

logger = logging.getLogger(__name__)


class ReupPipeline:
    """Coordinate the full auto-reup video workflow."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.project_root = Path(__file__).resolve().parent
        self.config_path = config_path or str(self.project_root / "config" / "settings.yaml")
        self.settings = self._load_settings()
        self.app_config = self.settings.get("app", {})

        self.downloader = VideoDownloader(config_path=self.config_path)
        self.transcriber = WhisperTranscriber(config_path=self.config_path, device="cpu")
        self.rewriter = LLMRewriter(config_path=self.config_path)
        self.translator = TranslationEngine(config_path=self.config_path)
        self.tts_engine = TTSEngine(config_path=self.config_path)
        self.subtitle_renderer = SubtitleRenderer(config_path=self.config_path)
        self.ffmpeg_processor = FFmpegProcessor(config_path=self.config_path)
        self.inpainter = VideoInpainter(config_path=self.config_path)
        self.video_upscaler = VideoUpscaler(config_path=self.config_path)

        initialize_workspace(clear_existing=True)
        self.temp_dir = TEMP_DIR
        self.output_dir = OUTPUT_DIR
        self.cleaned_dir = CLEANED_DIR
        self.input_dir = INPUT_DIR
        self.cleaned_video_path = CLEANED_VIDEO_PATH
        self.final_video_path = FINAL_VIDEO_PATH
        self.subtitle_srt_path = SUBTITLE_SRT_PATH
        self.subtitle_ass_path = SUBTITLE_ASS_PATH
        self.audio_output_path = AUDIO_OUTPUT_PATH
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self._clean_temp_files()

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

    def _run_gpu_inpainting_task(self, original_video: Path) -> Path:
        """Perform YOLO detection and ProPainter cleaning on the GPU thread."""
        self.cleaned_video_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            detected_boxes = self.inpainter.detect_watermark_and_text(str(original_video))
            if detected_boxes:
                logger.info(
                    "[INFO] Detected %s object frames requiring inpainting. Building masks and invoking ProPainter...",
                    len(detected_boxes),
                )
                try:
                    cleaned_video_path = self.inpainter.clean_video(
                        str(original_video),
                        str(self.cleaned_video_path),
                        detected_boxes=detected_boxes,
                        subvideo_length=min(self.app_config.get("propainter_subvideo_length", 30), 20),
                        raft_iter=self.app_config.get("propainter_raft_iter", 10),
                        resize_max_side=min(self.app_config.get("propainter_resize_max_side", 1280), 848),
                        fp16=self.app_config.get("propainter_fp16", True),
                        enable_vram_cleanup=self.app_config.get("propainter_enable_vram_cleanup", True),
                    )
                except Exception as exc:
                    logger.warning("[WARNING] GPU inpainting task failed (%s). Falling back to original video.", exc)
                    cleaned_video_path = str(self.cleaned_video_path)
                    shutil.copy2(str(original_video), str(cleaned_video_path))

                cleaned_path = Path(cleaned_video_path)
                if cleaned_path.exists() and cleaned_path.is_file():
                    return cleaned_path
                return Path(resolve_workspace_media_file(cleaned_video_path, expected_suffix=".mp4"))

            logger.info("[INFO] No watermark or text objects found. Copying original video to cleaned_video.mp4")
            shutil.copy2(str(original_video), str(self.cleaned_video_path))
            return self.cleaned_video_path
        except Exception as exc:
            logger.warning("[WARNING] GPU worker hit an error (%s). Falling back to original video.", exc)
            self.cleaned_video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(original_video), str(self.cleaned_video_path))
            return self.cleaned_video_path
        finally:
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass

    def _run_cpu_audio_subtitle_task(
        self,
        original_video: Path,
        target_language: str,
        api_target_language: str,
        auto_rewrite: bool,
        openrouter_api_key: Optional[str],
        tts_engine_mode: Optional[str],
        tts_mode: Optional[str],
        reference_audio_path: Optional[str],
        custom_voice: Optional[str],
        voice_preset: Optional[str],
        subtitle_mode: str,
        subtitle_font: Optional[str],
        subtitle_size: int,
        subtitle_color: str,
        subtitle_outline_color: str,
        subtitle_position: str,
    ) -> dict:
        """Extract audio/transcribe/translate/create TTS and subtitle files on the CPU thread."""
        transcript_data = self.transcriber.transcribe(str(original_video))
        original_segments = [
            {
                "start": float(segment.get("start", 0.0)),
                "end": float(segment.get("end", segment.get("start", 0.0))),
                "text": str(segment.get("text", "")).strip(),
            }
            for segment in transcript_data.get("segments", [])
            if str(segment.get("text", "")).strip()
        ]

        if not original_segments:
            raise RuntimeError("No subtitle or transcription text was extracted from the audio.")

        subtitle_mode = subtitle_mode.title()
        subtitle_language = self.sanitize_lang_code(target_language)
        translated_segments: list[dict] = []
        if subtitle_mode in {"Translated", "Dual"} or (tts_mode == "Translated narration"):
            logger.info("[INFO] Step 3/6: Translating OCR text to %s...", subtitle_language)
            translated_segments = self._translate_segments(original_segments, subtitle_language, openrouter_api_key)

        if subtitle_mode == "Translated" and translated_segments:
            subtitle_segments = translated_segments
        elif subtitle_mode == "Dual" and translated_segments:
            subtitle_segments = [
                {
                    "start": orig["start"],
                    "end": orig["end"],
                    "text": f"{orig['text']} | {trans['text']}",
                }
                for orig, trans in zip(original_segments, translated_segments)
            ]
        else:
            subtitle_segments = original_segments

        subtitle_style = {
            "font": subtitle_font or "Arial",
            "size": str(subtitle_size),
            "color": subtitle_color or "#FFFFFF",
            "border": subtitle_outline_color or "#000000",
            "shadow": subtitle_outline_color or "#000000",
        }

        self._write_srt_file(subtitle_segments, self.subtitle_srt_path)
        subtitle_source_path = self.subtitle_srt_path
        try:
            subtitle_source_path = find_file_anywhere(self.temp_dir, ".srt")
        except FileNotFoundError:
            subtitle_source_path = self.subtitle_srt_path
        if subtitle_source_path != self.subtitle_srt_path:
            subtitle_source_path = promote_file_to_destination(subtitle_source_path, self.subtitle_srt_path, search_root=self.temp_dir, move=True)

        self.subtitle_renderer.write_ass_file(str(self.subtitle_srt_path), str(self.subtitle_ass_path), subtitle_style)

        narration_text: Optional[str] = None
        if auto_rewrite:
            try:
                self.rewriter.set_api_key(openrouter_api_key)
                narration_text = self.rewriter.rewrite(transcript_data.get("full_text", ""), api_target_language)
            except Exception as exc:
                logger.warning("AI rewrite step failed: %s", exc)

        narration_text = narration_text or (
            " ".join(item["text"] for item in translated_segments)
            if tts_mode == "Translated narration" and translated_segments
            else transcript_data.get("full_text", "")
        )
        narration_text = narration_text.strip() or transcript_data.get("full_text", "")

        total_source_duration = sum(
            max(0.1, float(item.get("end", item.get("start", 0.0))) - float(item.get("start", 0.0)))
            for item in original_segments
        )

        logger.info("[INFO] Step 4/6: Generating narration audio...")
        if str(tts_engine_mode or "").lower().startswith("local"):
            asyncio.run(
                self.tts_engine.clone_speech(
                    narration_text,
                    str(self.audio_output_path),
                    reference_audio_path=reference_audio_path,
                    target_language=api_target_language,
                    voice=custom_voice,
                    voice_preset=voice_preset,
                    target_duration=total_source_duration,
                )
            )
        else:
            asyncio.run(
                self.tts_engine.generate_speech(
                    narration_text,
                    str(self.audio_output_path),
                    voice=custom_voice,
                    engine_mode="edge",
                    target_language=api_target_language,
                    target_duration=total_source_duration,
                )
            )

        discovered_audio_path: Path | None = None
        for extension in (".mp3", ".wav"):
            try:
                discovered_audio_path = find_file_anywhere(self.temp_dir, extension)
                break
            except FileNotFoundError:
                continue
        if discovered_audio_path is not None and discovered_audio_path != self.audio_output_path:
            discovered_audio_path = promote_file_to_destination(discovered_audio_path, self.audio_output_path, search_root=self.temp_dir, move=True)

        return {
            "audio_path": str(discovered_audio_path or self.audio_output_path),
            "subtitle_srt_path": str(self.subtitle_srt_path),
            "subtitle_ass_path": str(self.subtitle_ass_path),
            "subtitle_mode": subtitle_mode,
            "subtitle_style": subtitle_style,
        }

    def _format_srt_timestamp(self, seconds: float) -> str:
        """Convert seconds to SRT timestamp format."""
        total_milliseconds = int(seconds * 1000)
        hours, remainder = divmod(total_milliseconds, 3600 * 1000)
        minutes, remainder = divmod(remainder, 60 * 1000)
        seconds_part, milliseconds = divmod(remainder, 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds_part:02d},{milliseconds:03d}"

    def sanitize_lang_code(self, lang_input: Optional[object]) -> str:
        while isinstance(lang_input, (set, list, tuple)):
            if len(lang_input) > 0:
                lang_input = next(iter(lang_input))
            else:
                lang_input = "vi"
                break

        lang_str = str(lang_input or "vi").strip().lower()
        if "(" in lang_str:
            lang_str = lang_str.split("(")[0].strip()

        lang_map = {
            "zh": "zh-CN",
            "zh-cn": "zh-CN",
            "zh-tw": "zh-TW",
            "vi": "vi-VN",
            "en": "en-US",
            "ja": "ja-JP",
            "ko": "ko-KR",
        }
        return lang_map.get(lang_str, lang_str)

    def _normalize_language_for_api(self, language: Optional[object]) -> str:
        base_code = self.sanitize_lang_code(language)
        return {
            "vi": "vi-VN",
            "vi-vn": "vi-VN",
            "en": "en-US",
            "en-us": "en-US",
            "zh": "zh-CN",
            "zh-cn": "zh-CN",
            "zh-tw": "zh-TW",
            "ja": "ja-JP",
            "ja-jp": "ja-JP",
            "ko": "ko-KR",
            "ko-kr": "ko-KR",
            "th": "th-TH",
            "th-th": "th-TH",
        }.get(base_code, base_code)

    def _translate_segments(self, segments: Sequence[dict], target_language: str, api_key: Optional[str] = None) -> list[dict]:
        translated_segments: list[dict] = []
        try:
            translated_segments = self.translator.translate_segments(list(segments), target_language=target_language, api_key=api_key)
        except Exception as exc:
            logger.warning("Translation failed for segments: %s", exc)
            for segment in segments:
                text = str(segment.get("text", "")).strip()
                if not text:
                    continue
                translated_segments.append({
                    "start": float(segment.get("start", 0.0)),
                    "end": float(segment.get("end", segment.get("start", 0.0))),
                    "text": text,
                })
        return translated_segments

    def _remove_stale_output_path(self, output_path: Union[str, Path]) -> None:
        """Delete a stale file or directory that would block creating a subtitle output file."""
        safe_file_path(output_path)

    def _write_srt_file(self, segments: Sequence[dict], output_path: Path) -> Path:
        """Write transcription segments to a .srt file in the temp directory."""
        output_file = safe_file_path(output_path)

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
        tts_mode: Optional[str] = "Translated narration",
        reference_audio_path: Optional[str] = None,
        voice_preset: Optional[str] = None,
        subtitle_mode: str = "Original",
        inpaint_mode: str = "propainter",
        auto_detect_subtitles: bool = True,
        auto_remove_watermark: bool = True,
        subtitle_font: Optional[str] = "DejaVu Sans",
        subtitle_size: int = 32,
        subtitle_color: str = "#FFFFFF",
        subtitle_outline_color: str = "#000000",
        subtitle_position: str = "bottom",
        output_mode: str = "Keep original",
        enable_upscale: bool = False,
        upscale_factor: str = "2x (1080p Full HD)",
        propainter_subvideo_length: int = 30,
        propainter_raft_iter: int = 10,
        propainter_resize_max_side: int = 1280,
        propainter_fp16: bool = True,
        propainter_enable_vram_cleanup: bool = True,
        speed_factor: float = 1.05,
        hflip: bool = True,
        background_audio_path: Optional[str] = None,
    ) -> str:
        """Run the full video reup pipeline and return the output video path."""
        try:
            logger.info("[INFO] Starting pipeline for input: %s", input_source)
            self._remove_stale_output_path(self.subtitle_srt_path)

            target_language = self.sanitize_lang_code(target_language)
            api_target_language = self._normalize_language_for_api(target_language)

            video_path = self._resolve_input_file(input_source)
            processed_video = Path(video_path)
            original_segments: list[dict] = []

            logger.info("[INFO] Step 1/5: Starting parallel GPU and CPU tasks...")
            future_gpu = None
            future_cpu = None
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_gpu = executor.submit(
                    self._run_gpu_inpainting_task,
                    processed_video,
                )
                future_cpu = executor.submit(
                    self._run_cpu_audio_subtitle_task,
                    processed_video,
                    target_language,
                    api_target_language,
                    auto_rewrite,
                    openrouter_api_key,
                    tts_engine_mode,
                    tts_mode,
                    reference_audio_path,
                    custom_voice,
                    voice_preset,
                    subtitle_mode,
                    subtitle_font,
                    subtitle_size,
                    subtitle_color,
                    subtitle_outline_color,
                    subtitle_position,
                )

                cleaned_video_path = future_gpu.result()
                cpu_results = future_cpu.result()

            processed_video = cleaned_video_path
            audio_path = cpu_results["audio_path"]
            subtitle_ass_path = cpu_results["subtitle_ass_path"]

            logger.info("[INFO] Step 6/6: Final rendering with audio, subtitles, and ratio adjustments...")
            final_rendered = self.ffmpeg_processor.render_reup_video(
                video_path=str(processed_video),
                new_audio_path=str(audio_path),
                output_path=str(self.final_video_path),
                ass_path=str(subtitle_ass_path) if subtitle_ass_path else None,
                subtitle_text=None,
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

            final_output = final_rendered
            if enable_upscale:
                upscale_value = 2 if str(upscale_factor).startswith("2x") else 4
                upscale_output = self.output_dir / f"final_video_upscaled_{upscale_value}x.mp4"
                logger.info("[INFO] Upscaling final video to %sx...", upscale_value)
                final_output = self.video_upscaler.upscale(
                    input_video_path=str(final_rendered),
                    output_video_path=str(upscale_output),
                    scale=upscale_value,
                )

            if Path(final_output).resolve() != self.final_video_path.resolve():
                final_output = str(promote_file_to_destination(final_output, self.final_video_path, search_root=self.output_dir, move=True))

            self._clean_temp_files()
            absolute_output = str(Path(final_output).resolve())
            logger.info("[INFO] Pipeline completed successfully: %s", absolute_output)
            return absolute_output

        except Exception as exc:
            logger.exception("[ERROR] Pipeline failed at runtime")
            raise RuntimeError(f"Pipeline failed: {exc}") from exc
