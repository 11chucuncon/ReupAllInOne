from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Union

import yaml

from core.downloader import VideoDownloader
from core.ffmpeg_processor import FFmpegProcessor
from core.inpainter import VideoInpainter
from core.ocr_processor import OCRProcessor
from core.rewriter import LLMRewriter
from core.transcriber import WhisperTranscriber
from core.tts_engine import TTSEngine
from core.translation import TranslationEngine
from core.upscaler import VideoUpscaler
from core.subtitle_renderer import SubtitleRenderer

logger = logging.getLogger(__name__)


def sanitize_config(config: dict) -> dict:
    """Normalize raw pipeline config values and ensure safe string values."""
    clean_config: dict = {}
    for key, value in (config or {}).items():
        while isinstance(value, (set, tuple, list)):
            value = list(value)[0] if len(value) > 0 else ""
        clean_config[key] = value

    lang = str(clean_config.get("target_lang", "vi")).strip().lower()
    if "(" in lang:
        lang = lang.split("(")[0].strip()

    lang_map = {
        "zh": "zh-CN",
        "zh-cn": "zh-CN",
        "zh-tw": "zh-TW",
        "vi": "vi-VN",
        "en": "en-US",
        "ja": "ja-JP",
        "ko": "ko-KR",
    }
    clean_config["target_lang"] = lang_map.get(lang, "zh-CN" if "zh" in lang else lang)
    return clean_config


class ReupPipeline:
    """Coordinate the full auto-reup video workflow."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.project_root = Path(__file__).resolve().parent
        self.config_path = config_path or str(self.project_root / "config" / "settings.yaml")
        self.settings = self._load_settings()
        self.app_config = self.settings.get("app", {})

        self.downloader = VideoDownloader(config_path=self.config_path)
        self.transcriber = WhisperTranscriber(config_path=self.config_path)
        self.ocr_processor = OCRProcessor(config_path=self.config_path)
        self.rewriter = LLMRewriter(config_path=self.config_path)
        self.translator = TranslationEngine(config_path=self.config_path)
        self.tts_engine = TTSEngine(config_path=self.config_path)
        self.subtitle_renderer = SubtitleRenderer(config_path=self.config_path)
        self.ffmpeg_processor = FFmpegProcessor(config_path=self.config_path)
        self.inpainter = VideoInpainter(config_path=self.config_path)
        self.video_upscaler = VideoUpscaler(config_path=self.config_path)

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
        for segment in segments:
            text = str(segment.get("text", "")).strip()
            if not text:
                continue
            try:
                translated = self.translator.translate_text(text, target_language, api_key=api_key)
            except Exception as exc:
                logger.warning("Translation failed for segment: %s", exc)
                translated = text
            translated_segments.append({
                "start": float(segment.get("start", 0.0)),
                "end": float(segment.get("end", segment.get("start", 0.0))),
                "text": translated,
            })
        return translated_segments

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

    def process_video(self, config: dict) -> str:
        """Run the full video reup pipeline using a sanitized config dictionary."""
        try:
            config = sanitize_config(config or {})
            logger.info("[INFO] Starting pipeline with config: %s", config)

            target_language = config.get("target_lang", "vi")
            api_target_language = self._normalize_language_for_api(target_language)
            video_path = self._resolve_input_file(config.get("input_source"))
            processed_video = Path(video_path)
            original_segments: list[dict] = []

            auto_rewrite = bool(config.get("auto_rewrite", True))
            custom_voice = config.get("custom_voice")
            openrouter_api_key = config.get("openrouter_api_key")
            tts_engine_mode = config.get("tts_engine_mode", "Edge-TTS Free (Tốc độ cao)")
            tts_mode = config.get("tts_mode", "Translated narration")
            reference_audio_path = config.get("reference_audio_path")
            voice_preset = config.get("voice_preset")
            subtitle_mode = str(config.get("subtitle_mode", "Original"))
            inpaint_mode = str(config.get("inpaint_mode", "propainter"))
            auto_detect_subtitles = bool(config.get("auto_detect_subtitles", True))
            auto_remove_watermark = bool(config.get("auto_remove_watermark", True))
            subtitle_font = config.get("subtitle_font", "DejaVu Sans")
            subtitle_size = int(config.get("subtitle_size", 32) or 32)
            subtitle_color = config.get("subtitle_color", "#FFFFFF")
            subtitle_outline_color = config.get("subtitle_outline_color", "#000000")
            subtitle_position = str(config.get("subtitle_position", "bottom"))
            output_mode = str(config.get("output_mode", "Keep original"))
            enable_upscale = bool(config.get("enable_upscale", False))
            upscale_factor = str(config.get("upscale_factor", "2x (1080p Full HD)"))
            speed_factor = float(config.get("speed_factor", 1.05) or 1.05)
            hflip = bool(config.get("hflip", True))
            background_audio_path = config.get("background_audio_path")

            logger.info("[INFO] Step 1/6: Detecting on-screen text with OCR...")
            ocr_data = self.ocr_processor.detect_text(str(processed_video))
            original_segments = [
                {
                    "start": float(item.get("start_time", item.get("start", 0.0))),
                    "end": float(item.get("end_time", item.get("end", item.get("start", 0.0)))),
                    "text": str(item.get("text", "")).strip(),
                }
                for item in ocr_data.get("segments", [])
                if str(item.get("text", "")).strip()
            ]

            if not original_segments:
                logger.info("[INFO] No OCR segments were detected; falling back to audio transcription...")
                transcript_data = self.transcriber.transcribe(str(processed_video))
                original_segments = [
                    {
                        "start": float(segment.get("start", 0.0)),
                        "end": float(segment.get("end", segment.get("start", 0.0))),
                        "text": str(segment.get("text", "")).strip(),
                    }
                    for segment in transcript_data.get("segments", [])
                    if str(segment.get("text", "")).strip()
                ]
                ocr_data = transcript_data

            logger.info("[INFO] Step 2/6: Running video cleanup with inpainting mode '%s'...", inpaint_mode)
            output_clean = self.temp_dir / "cleaned_video.mp4"
            mask_video = None
            if auto_detect_subtitles or auto_remove_watermark:
                mask_image_path = self.temp_dir / "mask.png"
                mask_video = self.ocr_processor.build_mask_image(
                    str(processed_video),
                    str(mask_image_path),
                    original_segments,
                    auto_remove_watermark=auto_remove_watermark,
                )
            processed_video = Path(
                self.inpainter.clean_video(
                    str(processed_video),
                    str(output_clean),
                    mode=inpaint_mode,
                    mask_video_path=mask_video,
                )
            )

            if not original_segments:
                logger.info("[INFO] No OCR segments were detected; falling back to audio transcription...")
                transcript_data = self.transcriber.transcribe(str(processed_video))
                original_segments = [
                    {
                        "start": float(segment.get("start", 0.0)),
                        "end": float(segment.get("end", segment.get("start", 0.0))),
                        "text": str(segment.get("text", "")).strip(),
                    }
                    for segment in transcript_data.get("segments", [])
                    if str(segment.get("text", "")).strip()
                ]
                ocr_data = transcript_data

            if not original_segments:
                raise RuntimeError("No text was extracted by OCR or transcription.")

            subtitle_mode = subtitle_mode.title()
            subtitle_language = self.sanitize_lang_code(target_language)
            translated_segments = []
            if subtitle_mode in {"Translated", "Dual"} or (tts_mode == "Translated narration"):
                logger.info("[INFO] Step 3/6: Translating OCR text to %s...", subtitle_language)
                if auto_rewrite:
                    try:
                        self.rewriter.set_api_key(openrouter_api_key)
                        translated_segments = self.rewriter.rewrite_segments(original_segments, api_target_language)
                    except Exception as exc:
                        logger.warning("AI segment rewrite failed, falling back to simple translation: %s", exc)
                if not translated_segments:
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

            subtitle_path = self.temp_dir / "subtitle_overlay.srt"
            self._write_srt_file(subtitle_segments, subtitle_path)

            rewritten_text: Optional[str] = None
            if auto_rewrite:
                try:
                    self.rewriter.set_api_key(openrouter_api_key)
                    rewritten_text = self.rewriter.rewrite(ocr_data.get("full_text", ""), api_target_language)
                except Exception as exc:
                    logger.warning("AI rewrite step failed: %s", exc)

            narration_text = rewritten_text or (
                " ".join(item["text"] for item in translated_segments)
                if tts_mode == "Translated narration" and translated_segments
                else ocr_data.get("full_text", "")
            )
            narration_text = narration_text.strip() or ocr_data.get("full_text", "")

            audio_output_path = self.temp_dir / "new_voice.mp3"
            logger.info("[INFO] Step 4/6: Generating narration audio...")
            if str(tts_engine_mode or "").lower().startswith("local"):
                asyncio.run(
                    self.tts_engine.clone_speech(
                        narration_text,
                        str(audio_output_path),
                        reference_audio_path=reference_audio_path,
                        target_language=api_target_language,
                        voice=custom_voice,
                        voice_preset=voice_preset,
                    )
                )
            else:
                asyncio.run(
                    self.tts_engine.generate_speech(
                        narration_text,
                        str(audio_output_path),
                        voice=custom_voice,
                        engine_mode="edge",
                        target_language=api_target_language,
                    )
                )

            try:
                audio_duration = self.ffmpeg_processor.get_media_duration(str(audio_output_path))
                target_duration = original_segments[-1]["end"] if original_segments else None
                if target_duration and audio_duration > 0 and abs(audio_duration - target_duration) > 0.05:
                    adjusted_audio_path = self.temp_dir / "new_voice_timesync.mp3"
                    self.ffmpeg_processor.adjust_audio_speed(
                        str(audio_output_path),
                        str(adjusted_audio_path),
                        audio_duration / target_duration,
                    )
                    audio_output_path = adjusted_audio_path
            except Exception as exc:
                logger.warning("Audio time-sync adjustment failed: %s", exc)

            logger.info("[INFO] Step 5/6: Burning subtitles into video...")
            styled_video_path = self.temp_dir / "styled_video.mp4"
            subtitle_style = {
                "font": subtitle_font or "Arial",
                "size": str(subtitle_size),
                "color": subtitle_color or "#FFFFFF",
                "border": subtitle_outline_color or "#000000",
                "shadow": subtitle_outline_color or "#000000",
            }
            rendered_subtitle_video = self.subtitle_renderer.render_subtitles(
                str(processed_video),
                str(styled_video_path),
                str(subtitle_path),
                subtitle_mode.lower(),
                subtitle_style,
            )

            output_video_path = self.output_dir / f"reup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            logger.info("[INFO] Step 6/6: Final rendering with audio, speed, and ratio adjustments...")
            final_rendered = self.ffmpeg_processor.render_reup_video(
                video_path=rendered_subtitle_video,
                new_audio_path=str(audio_output_path),
                output_path=str(output_video_path),
                srt_path=None,
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
                upscale_output = self.output_dir / f"reup_{datetime.now().strftime('%Y%m%d_%H%M%S')}_upscaled_{upscale_value}x.mp4"
                logger.info("[INFO] Upscaling final video to %sx...", upscale_value)
                final_output = self.video_upscaler.upscale(
                    input_video_path=str(final_rendered),
                    output_video_path=str(upscale_output),
                    scale=upscale_value,
                )

            self._clean_temp_files()
            absolute_output = str(Path(final_output).resolve())
            logger.info("[INFO] Pipeline completed successfully: %s", absolute_output)
            return absolute_output

        except Exception as exc:
            logger.exception("[ERROR] Pipeline failed at runtime")
            raise RuntimeError(f"Pipeline failed: {exc}") from exc
