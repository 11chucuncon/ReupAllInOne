from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict

WORK_DIR = Path(os.environ.get("WORKSPACE_DIR", "/content/workspace")).expanduser().resolve()
INPUT_DIR = WORK_DIR / "input"
TEMP_DIR = WORK_DIR / "temp"
CLEANED_DIR = WORK_DIR / "cleaned"
OUTPUT_DIR = WORK_DIR / "output"

CLEANED_VIDEO_PATH = CLEANED_DIR / "cleaned_video.mp4"
FINAL_VIDEO_PATH = OUTPUT_DIR / "final_video.mp4"
SUBTITLE_SRT_PATH = TEMP_DIR / "subtitle_overlay.srt"
SUBTITLE_ASS_PATH = TEMP_DIR / "subtitle_overlay.ass"
AUDIO_OUTPUT_PATH = TEMP_DIR / "new_voice.mp3"


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}


def initialize_workspace(clear_existing: bool = True) -> Dict[str, Path]:
    """Create and optionally clear the centralized workspace directories."""
    workspace_dirs = {
        "WORK_DIR": WORK_DIR,
        "INPUT_DIR": INPUT_DIR,
        "TEMP_DIR": TEMP_DIR,
        "CLEANED_DIR": CLEANED_DIR,
        "OUTPUT_DIR": OUTPUT_DIR,
        "CLEANED_VIDEO_PATH": CLEANED_VIDEO_PATH,
        "FINAL_VIDEO_PATH": FINAL_VIDEO_PATH,
        "SUBTITLE_SRT_PATH": SUBTITLE_SRT_PATH,
        "SUBTITLE_ASS_PATH": SUBTITLE_ASS_PATH,
        "AUDIO_OUTPUT_PATH": AUDIO_OUTPUT_PATH,
    }

    for path in workspace_dirs.values():
        if clear_existing and path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.mkdir(parents=True, exist_ok=True)

    return workspace_dirs


def resolve_workspace_media_file(target_path: str | Path, expected_suffix: str = ".mp4") -> Path:
    """Ensure the target path resolves to a real media file inside the workspace."""
    path = Path(target_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.is_file():
        if path.suffix.lower() == expected_suffix:
            return path
        raise RuntimeError(f"Expected a media file with suffix {expected_suffix}, got {path}")

    if path.exists() and path.is_dir():
        candidates = sorted(
            item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS
        )
        if not candidates:
            raise RuntimeError(f"Workspace directory contains no media files: {path}")

        preferred = next((item for item in candidates if item.suffix.lower() == expected_suffix), candidates[0])
        resolved_output = path.with_suffix(expected_suffix) if path.suffix.lower() != expected_suffix else path / preferred.name
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        if resolved_output.exists() and resolved_output.is_dir():
            shutil.rmtree(resolved_output)
        shutil.copy2(str(preferred), str(resolved_output))
        return resolved_output

    raise FileNotFoundError(f"Workspace media file not found: {path}")
