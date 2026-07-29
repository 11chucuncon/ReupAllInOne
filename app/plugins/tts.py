from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from app.plugins.base import BaseStep


def _speak_to_file_with_pyttsx3(*args, **kwargs):
    if args and hasattr(args[0], "__class__") and not isinstance(args[0], (str, Path)):
        output_path = args[1]
        text = args[2]
    else:
        output_path = args[0] if args else kwargs.get("output_path")
        text = args[1] if len(args) > 1 else kwargs.get("text")

    output_path = Path(output_path)
    import pyttsx3

    engine = pyttsx3.init()
    engine.save_to_file(text, str(output_path))
    engine.runAndWait()


class TTSStep(BaseStep):
    name = "TTS"

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        config = self.config or {}
        engine = config.get("engine", "edge-tts")
        voice = config.get("voice", "default")
        speed = config.get("speed", 1.0)
        output_dir = Path(config.get("output_dir", "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_path = output_dir / "tts.wav"

        text = context.get("translation_result", {}).get("translated_text") or context.get("asr_result", {}).get("text", "")

        tts_result: Dict[str, Any] = {
            "engine": engine,
            "voice": voice,
            "speed": speed,
            "audio_path": str(audio_path),
            "text": text,
            "status": "prepared",
        }

        try:
            if engine == "pyttsx3":
                _speak_to_file_with_pyttsx3(self, audio_path, text)
                tts_result.update({"status": "generated", "engine_note": "pyttsx3 TTS"})
            elif engine == "xtts":
                tts_result.update({"status": "prepared", "engine_note": "Use XTTS for custom voice cloning"})
            else:
                audio_path.write_bytes(b"")
                tts_result.update({"status": "prepared", "engine_note": "Use edge-tts for fast multilingual TTS"})
        except Exception as exc:
            tts_result.update({"status": "failed", "engine_note": f"TTS backend error: {exc}"})

        context["tts_result"] = tts_result
        return context
