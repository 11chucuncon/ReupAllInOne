from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict

from app.plugins.base import BaseStep


def _candidate_ffmpeg_paths() -> list[str]:
    candidates = []
    for env_name in ("FFMPEG_BIN", "FFMPEG_PATH"):
        value = os.getenv(env_name)
        if value:
            candidates.append(value)

    if os.name == "nt":
        candidates.extend([
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        ])
    else:
        candidates.extend([
            "/usr/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "/opt/conda/bin/ffmpeg",
            "/usr/local/opt/ffmpeg/bin/ffmpeg",
        ])

    try:
        repo_root = Path(__file__).resolve().parents[2]
        candidates.extend([
            str(repo_root / "ffmpeg_bin" / "ffmpeg-master-latest-win64-gpl" / "bin" / "ffmpeg.exe"),
            str(repo_root / "ffmpeg_bin" / "ffmpeg-master-latest-win64-gpl" / "bin" / "ffmpeg"),
        ])
    except Exception:
        pass

    return candidates


class FFmpegStep(BaseStep):
    name = "FFmpeg Render"

    def _build_subtitle_file(self, context: Dict[str, Any], output_dir: Path) -> Path | None:
        asr_result = context.get("asr_result") or {}
        translation_result = context.get("translation_result") or {}
        timestamps = asr_result.get("timestamps") or []
        subtitle_style = context.get("subtitle_style") or {}
        subtitle_mode = (context.get("subtitle_mode") or "translated").lower()

        source_text = asr_result.get("text") or ""
        translated_text = translation_result.get("translated_text") or ""
        text = translated_text if subtitle_mode == "translated" else source_text
        if not text and subtitle_mode == "translated":
            text = source_text
        if not text and subtitle_mode == "source":
            text = translated_text

        if not timestamps and not text:
            return None

        subtitle_path = output_dir / "subtitles.srt"
        lines = []
        if timestamps:
            for index, item in enumerate(timestamps, start=1):
                start = item.get("start", 0.0)
                end = item.get("end", start + 2.0)
                caption_text = item.get("text") or text or ""
                start_ts = self._format_srt_time(start)
                end_ts = self._format_srt_time(end)
                lines.append(str(index))
                lines.append(f"{start_ts} --> {end_ts}")
                lines.append(caption_text)
                lines.append("")
        elif text:
            lines = ["1", "00:00:00,000 --> 00:00:02,000", text, ""]

        subtitle_path.write_text("\n".join(lines), encoding="utf-8")
        context["subtitle_file"] = str(subtitle_path)
        context["subtitle_style"] = subtitle_style
        return subtitle_path

    def _format_srt_time(self, seconds: float) -> str:
        total_seconds = max(int(seconds), 0)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},000"

    def _alignment_code(self, position: str) -> int:
        position_map = {
            "top": 2,
            "bottom": 8,
            "center": 5,
            "top_left": 1,
            "bottom_left": 7,
            "top_right": 3,
            "bottom_right": 9,
        }
        return position_map.get(position.lower(), 8)

    def _resolve_input_path(self, input_path: Path) -> Path:
        if input_path.exists():
            return input_path

        candidates = []
        if input_path.parent.exists():
            candidates.extend(
                [
                    input_path.with_suffix(".webm"),
                    input_path.with_suffix(f"{input_path.suffix}.webm") if input_path.suffix else input_path,
                    input_path.parent / f"{input_path.name}.webm",
                ]
            )
            candidates.extend(sorted(input_path.parent.glob(f"{input_path.stem}*")))

        for candidate in candidates:
            if isinstance(candidate, Path) and candidate.exists() and candidate.is_file():
                return candidate

        return input_path

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        input_path = self._resolve_input_path(Path(context.get("video_path", "downloads/raw.mp4")))
        output_dir = Path(context.get("output_dir", "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "final.mp4"

        if not input_path.exists():
            context["rendered_path"] = str(output_path)
            context["render_status"] = "failed"
            context["render_error"] = f"Input file not found: {input_path}"
            return context

        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            for candidate in _candidate_ffmpeg_paths():
                if Path(candidate).exists():
                    ffmpeg_path = candidate
                    break

        if not ffmpeg_path:
            context["rendered_path"] = str(output_path)
            context["render_status"] = "failed"
            context["render_error"] = "ffmpeg executable not found on PATH or common install locations"
            return context

        subtitle_path = self._build_subtitle_file(context, output_dir)
        cmd = [
            ffmpeg_path,
            "-y",
            "-i",
            str(input_path),
        ]
        if subtitle_path is not None:
            subtitle_style = context.get("subtitle_style") or {}
            font = subtitle_style.get("font", "Arial")
            font_size = subtitle_style.get("font_size", 48)
            color = subtitle_style.get("color", "white")
            position = subtitle_style.get("position", "bottom")
            outline_color = subtitle_style.get("outline_color", "black")
            outline_width = subtitle_style.get("outline_width", 2)
            style = subtitle_style.get("style", "default")

            force_style = (
                f"FontName={font},"
                f"FontSize={font_size},"
                f"PrimaryColour={color},"
                f"OutlineColour={outline_color},"
                f"Outline={outline_width},"
                f"Alignment={self._alignment_code(position)},"
                f"MarginV=20"
            )
            if style == "bold":
                force_style += ",Bold=1"
            elif style == "italic":
                force_style += ",Italic=1"

            subtitle_arg = subtitle_path.name
            cmd.extend(["-vf", f"subtitles={subtitle_arg}:force_style='{force_style}'"])
        cmd.extend([
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(output_path.name),
        ])

        try:
            completed = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(output_dir.resolve()),
            )
            if output_path.exists():
                context["rendered_path"] = str(output_path)
                context["render_status"] = "success"
                context["render_log"] = completed.stdout[-4000:] if completed.stdout else ""
            else:
                context["rendered_path"] = str(output_path)
                context["render_status"] = "failed"
                context["render_error"] = "ffmpeg completed but output file was not created"
        except subprocess.CalledProcessError as exc:
            context["rendered_path"] = str(output_path)
            context["render_status"] = "failed"
            context["render_error"] = exc.stdout[-4000:] if exc.stdout else str(exc)
            context["render_command"] = " ".join(cmd)
        except Exception as exc:
            context["rendered_path"] = str(output_path)
            context["render_status"] = "failed"
            context["render_error"] = str(exc)

        return context
