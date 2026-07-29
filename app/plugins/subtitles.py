from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from app.plugins.base import BaseStep
import wave
from pathlib import Path


def _secs_to_srt(ts: float) -> str:
    ms = int((ts - int(ts)) * 1000)
    h = int(ts // 3600)
    m = int((ts % 3600) // 60)
    s = int(ts % 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _translate_text(text: str, target: str) -> str:
    # Minimal pluggable translator: real models can be integrated here.
    if not target or target == "auto":
        return text
    # Preference order: OpenAI (if key), DeepL (if key), deep_translator fallback
    import os

    provider = os.environ.get("SUBTITLE_TRANSLATOR_PROVIDER", "auto").lower()

    # Try OpenAI chat completion for high-quality contextual translation
    try:
        if provider in ("openai", "auto") and os.environ.get("OPENAI_API_KEY"):
            import openai

            openai.api_key = os.environ.get("OPENAI_API_KEY")
            system = {
                "role": "system",
                "content": "You are a professional translator. Translate the user's text into the target language exactly, preserving meaning and punctuation. Respond with only the translated text."
            }
            user = {"role": "user", "content": f"Translate to {target}: {text}"}
            try:
                resp = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[system, user], max_tokens=2000)
                t = resp.choices[0].message.content.strip()
                if t:
                    return t
            except Exception:
                pass
    except Exception:
        pass

    # Try DeepL if configured
    try:
        if provider in ("deepl", "auto") and (os.environ.get("DEEPL_AUTH_KEY") or os.environ.get("DEEPL_API_KEY")):
            try:
                import deepl

                auth_key = os.environ.get("DEEPL_AUTH_KEY") or os.environ.get("DEEPL_API_KEY")
                translator = deepl.Translator(auth_key)
                translated = translator.translate_text(text, target_lang=target.upper())
                return translated.text
            except Exception:
                pass
    except Exception:
        pass

    # Fallback: deep_translator GoogleTranslator if installed
    try:
        from deep_translator import GoogleTranslator

        return GoogleTranslator(source="auto", target=target).translate(text)
    except Exception:
        return f"[{target}] " + text


def _translate_segments(segments: List[Dict[str, Any]], target: str, context_window: int = 1) -> List[str]:
    """Translate a list of segments (dicts with 'text') preserving order.
    Attempts OpenAI chat per-segment with neighboring context, then DeepL, then deep_translator fallback.
    Returns list of translated strings aligned with segments.
    """
    import os

    if not target or target == "auto":
        return [s.get("text", "") for s in segments]

    provider = os.environ.get("SUBTITLE_TRANSLATOR_PROVIDER", "auto").lower()

    # Try OpenAI per-segment with context
    if provider in ("openai", "auto") and os.environ.get("OPENAI_API_KEY"):
        try:
            import openai

            openai.api_key = os.environ.get("OPENAI_API_KEY")
            system = {
                "role": "system",
                "content": "You are a professional translator. Translate the user's text into the target language exactly, preserving meaning and punctuation. Keep translations concise and suitable for subtitle display. Respond with only the translated text."
            }
            results: List[str] = []
            for i, seg in enumerate(segments):
                before = " ".join([s.get("text", "") for s in segments[max(0, i - context_window):i]])
                after = " ".join([s.get("text", "") for s in segments[i + 1:i + 1 + context_window]])
                user_content = (
                    f"Prior context: {before}\nSegment: {seg.get('text','')}\nNext context: {after}\n\nTranslate the 'Segment' into {target}:"
                )
                user = {"role": "user", "content": user_content}
                try:
                    resp = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[system, user], max_tokens=800)
                    t = resp.choices[0].message.content.strip()
                    results.append(t if t else f"[{target}] " + seg.get("text", ""))
                except Exception:
                    results.append(f"[{target}] " + seg.get("text", ""))
            return results
        except Exception:
            pass

    # Try DeepL per-segment
    if provider in ("deepl", "auto") and (os.environ.get("DEEPL_AUTH_KEY") or os.environ.get("DEEPL_API_KEY")):
        try:
            import deepl

            auth_key = os.environ.get("DEEPL_AUTH_KEY") or os.environ.get("DEEPL_API_KEY")
            translator = deepl.Translator(auth_key)
            out: List[str] = []
            for seg in segments:
                try:
                    t = translator.translate_text(seg.get("text", ""), target_lang=target.upper())
                    out.append(t.text)
                except Exception:
                    out.append(f"[{target}] " + seg.get("text", ""))
            return out
        except Exception:
            pass

    # Fallback to deep_translator
    try:
        from deep_translator import GoogleTranslator

        gt = GoogleTranslator(source="auto", target=target)
        return [gt.translate(s.get("text", "")) for s in segments]
    except Exception:
        return [f"[{target}] " + s.get("text", "") for s in segments]


class SubtitleStep(BaseStep):
    name = "Subtitles"

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        config = self.config or {}
        # accept multiple possible keys for translation target
        translation_target = (
            context.get("translation_target")
            or context.get("translation_target_language")
            or context.get("target_language")
            or config.get("target")
        )
        asr = context.get("asr_result") or {}
        timestamps: List[Dict[str, Any]] = asr.get("timestamps", [])
        audio_path = asr.get("audio_path")
        preserve_timing = context.get("preserve_timing") or config.get("preserve_timing") or False

        # helper: get audio duration if available for clamping
        def _audio_duration(path: str | None) -> float | None:
            if not path:
                return None
            try:
                p = Path(path)
                if p.suffix.lower() == ".wav":
                    with wave.open(str(p), "rb") as wf:
                        frames = wf.getnframes()
                        rate = wf.getframerate()
                        return frames / float(rate) if rate else None
            except Exception:
                return None
            return None

        audio_duration = _audio_duration(audio_path)

        if not timestamps:
            context["subtitle_result"] = {"status": "no_timestamps", "srt_path": None}
            return context

        # prepare srt lines
        srt_lines: List[str] = []
        MIN_SEG_DUR = 0.4
        # If translation requested, translate segments in batch (preserve order)
        translated_texts: List[str] | None = None
        if translation_target:
            try:
                translated_texts = _translate_segments(timestamps, translation_target, context_window=1)
            except Exception:
                translated_texts = None

        for idx, seg in enumerate(timestamps, start=1):
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start + 4.0))
            # ensure minimal duration
            if end <= start + MIN_SEG_DUR:
                end = start + MIN_SEG_DUR
            # clamp to audio duration if known
            if audio_duration is not None and end > audio_duration:
                end = audio_duration
            if start >= end:
                start = max(0.0, end - MIN_SEG_DUR)
            text = seg.get("text", "")
            if translation_target:
                if translated_texts is not None and idx - 1 < len(translated_texts):
                    text = translated_texts[idx - 1]
                else:
                    text = _translate_text(text, translation_target)
            srt_lines.append(str(idx))
            srt_lines.append(f"{_secs_to_srt(start)} --> {_secs_to_srt(end)}")
            srt_lines.append(text)
            srt_lines.append("")

        # decide output path: save SRT next to audio file (or repo outputs)
        if audio_path:
            out_dir = Path(audio_path).parent
        else:
            out_dir = Path.cwd() / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        srt_name = (Path(audio_path).stem if audio_path else "subtitles")
        tgt = f".{translation_target}" if translation_target else ""
        srt_path = out_dir / f"{srt_name}{tgt}.srt"
        with open(srt_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(srt_lines))

        context["subtitle_result"] = {"status": "written", "srt_path": str(srt_path), "count": len(timestamps)}
        return context
