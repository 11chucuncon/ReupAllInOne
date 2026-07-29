from __future__ import annotations

from pathlib import Path
from typing import Optional


class VideoDownloader:
    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self.output_dir = output_dir or Path("outputs")

    def download(self, source_url: str, output_dir: Optional[Path] = None) -> str:
        target_dir = output_dir or self.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        output_path = target_dir / "source.mp4"
        output_path.write_bytes(b"mock video payload")
        return str(output_path)
