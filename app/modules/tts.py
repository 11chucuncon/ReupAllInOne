from __future__ import annotations

from pathlib import Path
from typing import List, Dict


class TTSProcessor:
    def synthesize(self, transcript: List[Dict[str, str]], output_dir: Path) -> str:
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_path = output_dir / "tts.wav"
        audio_path.write_bytes(b"mock tts audio")
        return str(audio_path)
