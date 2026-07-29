from __future__ import annotations

from pathlib import Path


class Renderer:
    def render(self, source_path: str, audio_path: str, output_dir: Path) -> str:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "final.mp4"
        output_path.write_bytes(b"mock rendered video")
        return str(output_path)
