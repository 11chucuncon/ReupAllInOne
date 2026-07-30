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

CLEANED_VIDEO_PATH = OUTPUT_DIR / "cleaned_video.mp4"
FINAL_VIDEO_PATH = OUTPUT_DIR / "final_video.mp4"
SUBTITLE_SRT_PATH = OUTPUT_DIR / "subtitles.srt"
SUBTITLE_ASS_PATH = OUTPUT_DIR / "subtitles.ass"
AUDIO_OUTPUT_PATH = OUTPUT_DIR / "new_voice.mp3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v"}


def safe_file_path(path_str: str | Path) -> Path:
    """Ensure a target output path is a real file path by removing any stale directory with the same name."""
    path = Path(path_str).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink(missing_ok=True)

    return path


def find_file_anywhere(search_dir: str | Path, extension: str) -> Path:
    """Recursively find the latest modified file with the requested extension inside a directory tree."""
    normalized_extension = extension if extension.startswith(".") else f".{extension}"
    search_path = Path(search_dir).expanduser().resolve()

    if search_path.is_file():
        if search_path.suffix.lower() == normalized_extension.lower():
            return search_path
        raise FileNotFoundError(f"No matching {normalized_extension} file found at {search_path}")

    if not search_path.exists():
        raise FileNotFoundError(f"Search directory does not exist: {search_path}")

    candidates = [
        candidate
        for candidate in search_path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() == normalized_extension.lower()
    ]
    if not candidates:
        raise FileNotFoundError(f"No matching {normalized_extension} file found under {search_path}")

    return max(candidates, key=os.path.getmtime)


def promote_file_to_destination(
    source_file: str | Path,
    destination_path: str | Path,
    search_root: str | Path | None = None,
    move: bool = True,
) -> Path:
    """Move or copy a discovered file to the intended destination and clean up empty parent folders."""
    source_path = Path(source_file).expanduser().resolve()
    destination = Path(destination_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.is_dir():
        shutil.rmtree(destination, ignore_errors=True)
    elif destination.exists():
        destination.unlink(missing_ok=True)

    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError(f"Source file does not exist: {source_path}")

    if source_path != destination:
        if move:
            shutil.move(str(source_path), str(destination))
        else:
            shutil.copy2(str(source_path), str(destination))

    if search_root is not None:
        root_path = Path(search_root).expanduser().resolve()
        current_dir = source_path.parent
        while current_dir != root_path and current_dir.exists() and current_dir != current_dir.parent:
            try:
                current_dir.rmdir()
            except OSError:
                break
            current_dir = current_dir.parent

    return destination


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

    if path.exists() and path.is_file():
        if path.suffix.lower() == expected_suffix.lower():
            return path
        raise RuntimeError(f"Expected a media file with suffix {expected_suffix}, got {path}")

    if path.exists() and path.is_dir():
        try:
            preferred = find_file_anywhere(path, expected_suffix)
        except FileNotFoundError as exc:
            raise RuntimeError(f"Workspace directory contains no media files: {path}") from exc

        resolved_output = path.with_suffix(expected_suffix) if path.suffix.lower() != expected_suffix.lower() else path / preferred.name
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        if resolved_output.exists() and resolved_output.is_dir():
            shutil.rmtree(resolved_output)
        elif resolved_output.exists():
            resolved_output.unlink(missing_ok=True)
        shutil.copy2(str(preferred), str(resolved_output))
        return resolved_output

    if not path.exists():
        raise FileNotFoundError(f"Media file or directory does not exist: {path}")

    return path
