from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Tuple
import shutil
import subprocess

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


def _get_audio_duration(path: Path) -> float | None:
    try:
        # Prefer Python-native WAV duration for extracted WAV files
        if path.suffix.lower() == ".wav":
            try:
                import wave

                with wave.open(str(path), "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    return frames / float(rate) if rate else None
            except Exception:
                pass

        # Fallback: try ffprobe (if available)
        ffprobe = shutil.which("ffprobe")
        if ffprobe:
            cmd = [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)]
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
            return float(out)

        # Last resort: try imageio_ffmpeg's ffmpeg exe to probe via ffmpeg -i output parsing (not ideal)
        try:
            import imageio_ffmpeg as iioff

            ffmpeg_exe = iioff.get_ffmpeg_exe()
            if ffmpeg_exe:
                cmd = [ffmpeg_exe, "-i", str(path)]
                out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode()
                for line in out.splitlines():
                    if "Duration:" in line:
                        # parse Duration: 00:00:03.08
                        try:
                            part = line.split("Duration:", 1)[1].split(",", 1)[0].strip()
                            h, m, s = part.split(":")
                            secs = int(h) * 3600 + int(m) * 60 + float(s)
                            return secs
                        except Exception:
                            continue
        except Exception:
            pass

    except Exception:
        pass
    return None


def _approximate_segments_from_text(text: str, duration: float | None, target_segment_secs: float = 4.0) -> List[Dict[str, Any]]:
    words = text.split()
    if not words:
        return []
    total_words = len(words)
    if duration and duration > 0:
        # determine number of segments by target seconds
        import math

        n_segments = max(1, math.ceil(duration / target_segment_secs))
        # estimate words per second and words per segment
        wps = total_words / duration if duration and total_words else max(0.5, total_words / (n_segments * target_segment_secs))
        words_per_segment = max(1, int(round(wps * target_segment_secs)))
        segments = []
        idx = 0
        # map words to timestamps proportionally
        if duration and total_words:
            # compute time position per word
            time_per_word = duration / total_words
            for i in range(n_segments):
                if idx >= total_words:
                    break
                seg_word_count = min(words_per_segment, total_words - idx)
                start = idx * time_per_word
                end = (idx + seg_word_count) * time_per_word
                seg_words = words[idx: idx + seg_word_count]
                idx += seg_word_count
                segments.append({"start": float(start), "end": float(end), "text": " ".join(seg_words)})
            if idx < total_words:
                # append remaining words to last segment
                segments[-1]["text"] = segments[-1]["text"] + " " + " ".join(words[idx:])
        else:
            # fallback: split by words into roughly target word chunks
            words_per_segment = max(1, total_words // n_segments)
            idx = 0
            for i in range(n_segments):
                seg_words = words[idx: idx + words_per_segment]
                idx += words_per_segment
                segments.append({"start": float(i * target_segment_secs), "end": float((i + 1) * target_segment_secs), "text": " ".join(seg_words)})
            if idx < total_words:
                segments[-1]["text"] = segments[-1]["text"] + " " + " ".join(words[idx:])
        return segments
    else:
        # duration unknown: split approx every 10 words
        n_segments = max(1, total_words // 10)
        words_per_segment = max(1, total_words // n_segments)
        segments = []
        idx = 0
        for i in range(n_segments):
            seg_words = words[idx: idx + words_per_segment]
            idx += words_per_segment
            segments.append({"start": float(i * target_segment_secs), "end": float((i + 1) * target_segment_secs), "text": " ".join(seg_words)})
        if idx < total_words:
            segments[-1]["text"] = segments[-1]["text"] + " " + " ".join(words[idx:])
        return segments


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
                        # try to recover: keep text placeholder but attempt approximate timestamps
                        approx_ts: List[Dict[str, Any]] = []
                        try:
                            duration = _get_audio_duration(audio_file)
                            approx_ts = _approximate_segments_from_text("Whisper placeholder", duration)
                        except Exception:
                            approx_ts = []
                        asr_result.update({
                            "status": "transcribed",
                            "text": "Whisper placeholder",
                            "timestamps": approx_ts,
                            "engine_note": f"faster_whisper failed, using placeholder: {exc}",
                        })
                else:
                    # no faster_whisper: try whisper package or fallback to approximate segmentation
                    try:
                        import whisper as _wh

                        model = _wh.load_model("small")
                        wres = model.transcribe(str(audio_file), language=language if language != "auto" else None)
                        text = wres.get("text", "")
                        # whisper python package provides segments in some versions
                        segments = wres.get("segments") or []
                        timestamps = [{"start": float(s.get("start", 0.0)), "end": float(s.get("end", 0.0)), "text": (s.get("text") or "").strip()} for s in segments if s.get("text")]
                        if not timestamps:
                            duration = _get_audio_duration(audio_file)
                            timestamps = _approximate_segments_from_text(text or "", duration)
                        asr_result.update({
                            "status": "transcribed",
                            "text": text,
                            "timestamps": timestamps,
                            "engine_note": "whisper transcription",
                        })
                    except Exception:
                        duration = _get_audio_duration(audio_file)
                        approx_ts = _approximate_segments_from_text("Whisper placeholder", duration)
                        asr_result.update({
                            "status": "transcribed",
                            "text": "Whisper placeholder",
                            "timestamps": approx_ts,
                            "engine_note": "Use Whisper for general transcription (fallback)",
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
