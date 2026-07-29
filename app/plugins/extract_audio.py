from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

from app.plugins.base import BaseStep


def _resolve_ffmpeg_path() -> str | None:
    if shutil.which("ffmpeg"):
        return shutil.which("ffmpeg")

    for env_name in ("FFMPEG_BIN", "FFMPEG_PATH"):
        value = Path(os.getenv(env_name, ""))
        if value and value.exists():
            return str(value)

    if importlib.util.find_spec("imageio_ffmpeg") is not None:
        try:
            import imageio_ffmpeg

            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            if ffmpeg_exe and Path(ffmpeg_exe).exists():
                return str(ffmpeg_exe)
        except Exception:
            pass

    return None


class ExtractAudioStep(BaseStep):
    name = "Extract Audio"

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        input_path = Path(context.get("video_path") or "downloads/raw.mp4")
        output_dir = Path(context.get("output_dir", "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        audio_out = output_dir / "audio.wav"

        if not input_path.exists():
            context["audio_path"] = str(audio_out)
            context["audio_extraction_status"] = "failed"
            context["audio_extraction_error"] = f"Input file not found: {input_path}"
            return context

        ffmpeg_bin = _resolve_ffmpeg_path()
        if not ffmpeg_bin:
            context["audio_path"] = str(audio_out)
            context["audio_extraction_status"] = "failed"
            context["audio_extraction_error"] = "ffmpeg executable not found"
            return context

        cmd = [ffmpeg_bin, "-y", "-i", str(input_path), "-vn", "-ac", "1", "-ar", "16000", str(audio_out)]
        progress = context.get("progress_callback")
        try:
            # stream ffmpeg output if progress callback provided
            if callable(progress):
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in proc.stdout:
                    try:
                        progress(f"ffmpeg-extract:{line.rstrip()}")
                    except Exception:
                        pass
                proc.wait()
                ret = proc.returncode
                if ret == 0 and audio_out.exists():
                    context["audio_path"] = str(audio_out)
                    context["audio_extraction_status"] = "success"
                else:
                    context["audio_path"] = str(audio_out)
                    context["audio_extraction_status"] = "failed"
                    context["audio_extraction_error"] = f"ffmpeg exit code {ret}"
            else:
                completed = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if audio_out.exists():
                    context["audio_path"] = str(audio_out)
                    context["audio_extraction_status"] = "success"
                    context["audio_extraction_log"] = (completed.stdout or completed.stderr)[-2000:]
                else:
                    context["audio_path"] = str(audio_out)
                    context["audio_extraction_status"] = "failed"
                    context["audio_extraction_error"] = "ffmpeg completed but audio file not created"
        except subprocess.CalledProcessError as exc:
            context["audio_path"] = str(audio_out)
            context["audio_extraction_status"] = "failed"
            context["audio_extraction_error"] = exc.stderr[-2000:] if exc.stderr else str(exc)
        except Exception as exc:
            context["audio_path"] = str(audio_out)
            context["audio_extraction_status"] = "failed"
            context["audio_extraction_error"] = str(exc)

        return context
