from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable


def install_dependencies() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "gradio", "PyYAML", "yt-dlp"],
        check=True,
    )


def mount_drive() -> None:
    try:
        from google.colab import drive

        drive.mount("/content/drive", force_remount=False)
    except Exception as exc:
        print("Drive mount skipped or already mounted:", exc)


def find_repo_root() -> Path | None:
    candidates = [
        Path.cwd(),
        Path("/content/project"),
        Path("/content/video-pipeline"),
        Path("/content/drive/MyDrive/A"),
        Path("/content/drive/MyDrive/video-pipeline"),
        Path("/content/drive/My Drive/A"),
        Path("/content/drive/My Drive/video-pipeline"),
    ]
    for candidate in candidates:
        if (candidate / "app" / "plugins" / "runner.py").exists():
            return candidate
    return None


def clone_repo_if_needed() -> Path:
    repo_root = Path("/content/project")
    if (repo_root / "app" / "plugins" / "runner.py").exists():
        return repo_root

    if repo_root.exists():
        shutil.rmtree(repo_root, ignore_errors=True)

    for url in ["https://github.com/hanhwannau/A.git", "https://github.com/hanhwannau/A"]:
        for branch in ["master", "main"]:
            print(f"Trying {url} branch {branch}...")
            try:
                subprocess.run(
                    ["git", "clone", "--depth", "1", "--branch", branch, url, str(repo_root)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                print("Clone succeeded")
                return repo_root
            except subprocess.CalledProcessError as exc:
                print("Clone failed:", exc.returncode)
                print("stdout:", exc.stdout)
                print("stderr:", exc.stderr)

    raise RuntimeError("Failed to clone repository from GitHub. Check connectivity/authentication and whether the repo is public.")


def setup_repository() -> Path:
    repo_root = find_repo_root()
    if repo_root is None:
        repo_root = clone_repo_if_needed()
        repo_root = find_repo_root()

    if repo_root is None:
        raise FileNotFoundError(
            "Could not find the project repository. Please ensure the repo is available at /content/project or in Drive."
        )
    os.chdir(repo_root)
    sys.path.insert(0, str(repo_root))
    return repo_root


def run_video_pipeline(
    video_path: str,
    extract_audio: bool = False,
    generate_subtitles: bool = True,
    subtitle_mode: str = "translated",
    translation_target: str | None = None,
    asr_language: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[str, str, str]:
    repo_root = setup_repository()
    from app.plugins.runner import run_from_config

    input_path = Path(video_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    extra_context: dict = {}

    # If user requested extract_audio, enable the ExtractAudioStep in the pipeline
    if extract_audio:
        extra_context["enable_extract_audio_step"] = True

    # subtitle / translation options
    extra_context["subtitle_mode"] = (subtitle_mode or "translated").lower()
    if translation_target:
        extra_context["translation_target_language"] = translation_target
    if asr_language:
        extra_context["asr_language"] = asr_language

    result = run_from_config("config_pipeline_full.yaml", video_url=str(input_path.resolve()), extra_context=extra_context, progress_callback=progress_callback)
    rendered_path = Path(result.get("rendered_path", "outputs/final.mp4"))
    if not rendered_path.is_absolute():
        rendered_path = (repo_root / rendered_path).resolve()

    if not rendered_path.exists():
        raise FileNotFoundError(f"Rendered output not found: {rendered_path}")

    output_folder = Path("/content/drive/MyDrive/colab_outputs")
    if output_folder.exists() or output_folder.parent.exists():
        try:
            output_folder.mkdir(parents=True, exist_ok=True)
            saved_path = output_folder / rendered_path.name
            saved_path.write_bytes(rendered_path.read_bytes())
            output_message = f"Output copied to Drive: {saved_path}"
        except Exception as exc:
            output_message = f"Output saved locally at {rendered_path}, but could not copy to Drive: {exc}"
    else:
        output_message = f"Output saved locally at {rendered_path}"
    return result.get("render_status", "unknown"), str(rendered_path), output_message


def build_interface() -> None:
    import gradio as gr

    def extract_audio_only(video_file: str | Path | None):
        # returns: steps_log, ffmpeg_log, rendered_path, info_message, audio_path, run_log_path
        if video_file is None:
            return "Không có video", "", "", "Vui lòng upload video", "", None
        repo_root = setup_repository()
        input_path = Path(str(video_file))
        outputs_dir = repo_root / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        audio_out = outputs_dir / "audio.wav"
        try:
            subprocess.run([
                shutil.which("ffmpeg") or "ffmpeg",
                "-y",
                "-i",
                str(input_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                str(audio_out),
            ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            run_log = outputs_dir / f"progress-extract-{time.strftime('%Y%m%d-%H%M%S')}.log"
            try:
                with open(run_log, "a", encoding="utf-8") as fh:
                    fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Tách audio: {audio_out}\n")
            except Exception:
                run_log = None
            return "Tách audio xong", "", "", f"Audio extracted: {audio_out}", str(audio_out), str(run_log) if run_log is not None else None
        except Exception as exc:
            return "error", "", "", f"Audio extraction failed: {exc}", "", None

    def prettify(msg: str) -> str:
        STEP_NAME_MAP = {
            "Download Video": "Tải video",
            "ASR": "Nhận diện giọng nói (ASR)",
            "Translation": "Dịch văn bản",
            "TranslateStep": "Dịch (bước phụ)",
            "OCR": "Nhận diện chữ (OCR)",
            "TTSStep": "TTS (chuyển văn bản -> giọng nói)",
            "FFmpeg Render": "Kết xuất (ffmpeg)",
            "WatermarkStep": "Thêm watermark",
            "Extract Audio": "Tách audio",
        }
        try:
            if msg.startswith("start:"):
                step = msg.split(":", 1)[1]
                display = STEP_NAME_MAP.get(step, step)
                return f"Bắt đầu: {display}"
            if msg.startswith("end:"):
                step = msg.split(":", 1)[1]
                display = STEP_NAME_MAP.get(step, step)
                return f"Kết thúc: {display}"
            if msg.startswith("download:"):
                return f"Tải về: {msg.split(':',1)[1]}"
            if msg.startswith("asr:"):
                text = msg.split(":", 1)[1]
                return f"ASR: {text[:300]}"
            if msg.startswith("translate:"):
                text = msg.split(":", 1)[1]
                return f"Dịch: {text[:300]}"
            if msg.startswith("render:"):
                status = msg.split(":", 1)[1]
                if status.strip().lower() == "success":
                    return "Kết xuất: Thành công"
                return f"Kết xuất: {status}"
            if msg.startswith("ffmpeg-extract:"):
                return f"ffmpeg (tách audio): {msg.split(':',1)[1]}"
            if msg.startswith("ffmpeg:"):
                return f"ffmpeg: {msg.split(':',1)[1]}"
        except Exception:
            pass
        return msg

    def process_stream(
        video_file: str | Path | None,
        extract_audio: bool,
        generate_subtitles: bool,
        subtitle_mode: str,
        translation_target: str,
        asr_language: str,
    ):
        # generator that streams progress updates via a background thread
        if video_file is None:
            yield "Không có video", "", "", "", "", None
            return

        repo_root_local = None
        try:
            repo_root_local = setup_repository()
            (repo_root_local / "outputs").mkdir(parents=True, exist_ok=True)
        except Exception:
            repo_root_local = None

        # per-run logfile
        run_id = time.strftime('%Y%m%d-%H%M%S')
        run_log_path = None
        if repo_root_local is not None:
            run_log_path = repo_root_local / "outputs" / f"progress-{run_id}.log"
            try:
                with open(run_log_path, "a", encoding="utf-8") as fh:
                    fh.write(f"Run start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            except Exception:
                run_log_path = None

        events: list[str] = []
        step_lines: list[str] = []
        ffmpeg_lines: list[str] = []

        def progress_cb(message: str) -> None:
            events.append(message)
            # write prettified log to outputs/progress.log and per-run log if available
            try:
                pretty = prettify(message)
                timestamped = f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {pretty}\n"
                if repo_root_local is not None:
                    log_path = repo_root_local / "outputs" / "progress.log"
                    with open(log_path, "a", encoding="utf-8") as fh:
                        fh.write(timestamped)
                if run_log_path is not None:
                    with open(run_log_path, "a", encoding="utf-8") as fh:
                        fh.write(timestamped)
            except Exception:
                pass

        result_container: dict = {}

        def target() -> None:
            try:
                status, output_path, message = run_video_pipeline(
                    str(video_file),
                    extract_audio=extract_audio,
                    generate_subtitles=generate_subtitles,
                    subtitle_mode=subtitle_mode,
                    translation_target=translation_target or None,
                    asr_language=asr_language or None,
                    progress_callback=progress_cb,
                )
                result_container['result'] = (status, output_path, message)
            except Exception as exc:
                result_container['result'] = ('error', '', str(exc))

        thread = threading.Thread(target=target, daemon=True)
        thread.start()

        # stream events while thread runs; accumulate prettified log into step and ffmpeg panels
        while thread.is_alive() or events:
            while events:
                ev = events.pop(0)
                if ev.startswith("ffmpeg"):
                    ffmpeg_lines.append(ev.split(":", 1)[1])
                else:
                    step_lines.append(prettify(ev))
                # yield accumulated logs into the two textboxes
                yield "\n".join(step_lines), "\n".join(ffmpeg_lines), "", "", "", None
            time.sleep(0.2)

        # final result
        status, output_path, message = result_container.get('result', ('error', '', 'No result'))
        return_steps = "\n".join(step_lines)
        return_ffmpeg = "\n".join(ffmpeg_lines)
        run_log_str = str(run_log_path) if run_log_path is not None else None
        yield return_steps, return_ffmpeg, output_path, message, "", run_log_str

    with gr.Blocks() as demo:
        gr.Markdown('# Video Pipeline Colab UI')
        with gr.Row():
            video_input = gr.File(label='Upload video file', file_count='single', type='filepath')
        with gr.Row():
            extract_chk = gr.Checkbox(label='Extract audio (wav)', value=False)
            subs_chk = gr.Checkbox(label='Generate subtitles', value=True)
            subtitle_mode = gr.Radio(['translated', 'source'], label='Subtitle mode', value='translated')
        with gr.Row():
            translation_target = gr.Dropdown(['auto','en','vi','zh','fr','es'], label='Translation target language', value='vi')
            asr_lang = gr.Textbox(label='ASR language (e.g. en, vi) or "auto"', value='auto')
        with gr.Row():
            run_button = gr.Button('Run pipeline')
            extract_button = gr.Button('Extract audio only')
        with gr.Row():
            steps_out = gr.Textbox(label='Steps log', interactive=False, lines=12)
            ffmpeg_out = gr.Textbox(label='ffmpeg log', interactive=False, lines=12)
        with gr.Row():
            rendered_out = gr.Textbox(label='Rendered output path', interactive=False)
            info_out = gr.Textbox(label='Info / Drive copy status', interactive=False)
        with gr.Row():
            audio_out = gr.Textbox(label='Audio path (if extracted)', interactive=False)
        with gr.Row():
            log_file = gr.File(label='Download run log')

        run_button.click(
            fn=process_stream,
            inputs=[video_input, extract_chk, subs_chk, subtitle_mode, translation_target, asr_lang],
            outputs=[steps_out, ffmpeg_out, rendered_out, info_out, audio_out, log_file],
        )

        extract_button.click(
            fn=extract_audio_only,
            inputs=[video_input],
            outputs=[steps_out, ffmpeg_out, rendered_out, info_out, audio_out, log_file],
        )

    demo.launch(share=True)


if __name__ == '__main__':
    install_dependencies()
    mount_drive()
    build_interface()
