from __future__ import annotations

import os
import sys
from pathlib import Path

from app.plugins.runner import run_from_config

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if sys.stderr.encoding != 'utf-8':
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


def find_sample_video() -> Path:
    root = Path('.')
    candidates = []
    for ext in ['*.mp4', '*.webm', '*.mkv', '*.avi']:
        for path in root.rglob(ext):
            if 'outputs' in path.parts:
                continue
            candidates.append(path)
    # Prefer a known valid sample within downloads if available
    for candidate in candidates:
        if 'downloads' in candidate.parts and candidate.suffix.lower() == '.mp4':
            return candidate
    return candidates[0] if candidates else Path('')


def main() -> None:
    ffmpeg_exe = None
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        print('Could not locate imageio-ffmpeg executable:', exc)

    if ffmpeg_exe and Path(ffmpeg_exe).exists():
        ffmpeg_dir = str(Path(ffmpeg_exe).parent)
        os.environ['PATH'] = ffmpeg_dir + os.pathsep + os.environ.get('PATH', '')
        os.environ['FFMPEG_BIN'] = ffmpeg_exe
        print('Using ffmpeg at', ffmpeg_exe)
    else:
        print('ffmpeg executable not found. Ensure ffmpeg is installed or imageio-ffmpeg provides one.')

    sample_video = find_sample_video()
    print('Sample video:', sample_video)
    if not sample_video.exists():
        raise FileNotFoundError('No sample video found.')

    result = run_from_config(
        'config_pipeline_full.yaml',
        video_url=str(sample_video),
        extra_context={
            'enable_extract_audio_step': True,
            'subtitle_mode': 'translated',
            'translation_target_language': 'vi',
            'asr_language': 'auto',
        },
        progress_callback=lambda msg: print('PROG>', msg),
    )
    print('RESULT', repr(result))


if __name__ == '__main__':
    main()
