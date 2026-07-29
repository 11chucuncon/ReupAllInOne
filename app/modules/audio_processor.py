from __future__ import annotations

from pathlib import Path
from typing import Dict


class AudioProcessor:
    def process(self, source_path: str, output_dir: Path) -> Dict[str, str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        vocal_path = output_dir / "vocal.wav"
        instrumental_path = output_dir / "instrumental.wav"
        vocal_path.write_bytes(b"mock vocal")
        instrumental_path.write_bytes(b"mock instrumental")
        return {
            "vocal": str(vocal_path),
            "instrumental": str(instrumental_path),
        }
