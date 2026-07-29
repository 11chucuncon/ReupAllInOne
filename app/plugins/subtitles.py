from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from app.plugins.base import BaseStep


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
    try:
        # try to use any available translator library (example: transformers/argos)
        from deep_translator import GoogleTranslator

        return GoogleTranslator(source="auto", target=target).translate(text)
    except Exception:
        # fallback: mark text as translated but keep original
        return f"[{target}] " + text


class SubtitleStep(BaseStep):
    name = "Subtitles"

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        config = self.config or {}
        translation_target = context.get("translation_target") or config.get("target")
        asr = context.get("asr_result") or {}
        timestamps: List[Dict[str, Any]] = asr.get("timestamps", [])
        audio_path = asr.get("audio_path")

        if not timestamps:
            context["subtitle_result"] = {"status": "no_timestamps", "srt_path": None}
            return context

        # prepare srt lines
        srt_lines: List[str] = []
        for idx, seg in enumerate(timestamps, start=1):
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start + 4.0))
            text = seg.get("text", "")
            if translation_target:
                text = _translate_text(text, translation_target)
            srt_lines.append(str(idx))
            srt_lines.append(f"{_secs_to_srt(start)} --> {_secs_to_srt(end)}")
            srt_lines.append(text)
            srt_lines.append("")

        # decide output path
        out_dir = Path(audio_path).parent if audio_path else Path.cwd()
        out_dir = out_dir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        srt_name = (Path(audio_path).stem if audio_path else "subtitles")
        tgt = f".{translation_target}" if translation_target else ""
        srt_path = out_dir / f"{srt_name}{tgt}.srt"
        with open(srt_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(srt_lines))

        context["subtitle_result"] = {"status": "written", "srt_path": str(srt_path), "count": len(timestamps)}
        return context
