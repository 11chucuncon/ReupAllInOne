from __future__ import annotations

from pathlib import Path
from typing import List, Dict


class ASRProcessor:
    def transcribe(self, audio_path: str, output_dir: Path) -> List[Dict[str, str]]:
        output_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = output_dir / "transcript.json"
        transcript_path.write_text(
            '[{"start": "00:00:00", "end": "00:00:02", "text": "Hello world"}]',
            encoding="utf-8",
        )
        return [{"start": "00:00:00", "end": "00:00:02", "text": "Hello world"}]
