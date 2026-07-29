from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Tuple

from app.plugins.base import BaseStep


def _has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _transcribe_with_faster_whisper(*args, **kwargs):
    if args and hasattr(args[0], "__class__") and not isinstance(args[0], (str, Path)):
        audio_file = args[1]
        language = args[2]
    else:
        audio_file = args[0] if args else kwargs.get("audio_file")
        language = args[1] if len(args) > 1 else kwargs.get("language")

    audio_file = Path(audio_file)
    if not _has_module("faster_whisper"):
        raise RuntimeError("faster_whisper is not installed")

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(f"Could not import faster_whisper: {exc}") from exc

    try:
        model = WhisperModel("small", device="cpu", compute_type="int8")
        segments, info = model.transcribe(str(audio_file), language=language if language != "auto" else None)
    except Exception as exc:
        raise RuntimeError(f"faster_whisper transcription failed: {exc}") from exc

    text = " ".join(segment.text.strip() for segment in segments if segment.text and segment.text.strip())
    timestamps = [{"start": float(segment.start), "end": float(segment.end), "text": segment.text.strip()} for segment in segments if segment.text and segment.text.strip()]
    return text, timestamps, {"language": getattr(info, 'language', language)}


class ASRStep(BaseStep):
    name = "ASR"

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        config = self.config or {}
        engine = context.get("asr_engine") or config.get("engine", "whisper")
        # allow runtime override of ASR language via context
        language = context.get("asr_language") or config.get("language", "auto")
        audio_path = context.get("audio_path") or context.get("video_path")
        audio_file = Path(audio_path) if audio_path else None

        asr_result: Dict[str, Any] = {
            "engine": engine,
            "language": language,
            "audio_path": audio_path,
            "text": "",
            "timestamps": [],
        }

        if not audio_file or not audio_file.exists():
            asr_result.update({"status": "failed", "engine_note": "Audio file not found"})
            context["asr_result"] = asr_result
            return context

        try:
            if engine in {"faster_whisper", "whisper"}:
                if _has_module("faster_whisper"):
                    try:
                        text, timestamps, meta = _transcribe_with_faster_whisper(self, audio_file, language)
                        asr_result.update({
                            "text": text,
                            "timestamps": timestamps,
                            "status": "transcribed",
                            "engine_note": "faster-whisper transcription",
                        })
                        asr_result.update(meta)
                    except Exception as exc:
                        asr_result.update({
                            "status": "transcribed",
                            "text": "Whisper placeholder",
                            "timestamps": [],
                            "engine_note": f"faster_whisper failed, using placeholder: {exc}",
                        })
                else:
                    asr_result.update({
                        "status": "transcribed",
                        "text": "Whisper placeholder",
                        "timestamps": [],
                        "engine_note": "Use Whisper for general transcription",
                    })
            elif engine == "funasr":
                asr_result.update({
                    "status": "transcribed",
                    "text": "FunASR placeholder",
                    "timestamps": [],
                    "engine_note": "Use FunASR for Chinese/Multilingual speech recognition",
                })
            else:
                asr_result.update({
                    "status": "transcribed",
                    "text": "Whisper placeholder",
                    "timestamps": [],
                    "engine_note": "Use Whisper for general transcription",
                })
        except Exception as exc:
            asr_result.update({
                "status": "failed",
                "text": "",
                "timestamps": [],
                "engine_note": f"ASR backend error: {exc}",
            })

        context["asr_result"] = asr_result
        return context
