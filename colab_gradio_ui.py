from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
import threading
import time
from typing import Callable


def install_dependencies() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "gradio", "PyYAML", "yt-dlp"],
        check=True,
    )


def mount_drive() -> None:
    try:
        from google.colab import drive

        drive.mount('/content/drive', force_remount=False)
    except Exception as exc:
        print('Drive mount skipped or already mounted:', exc)

def setup_repository() -> Path:
    repo_root = find_repo_root()
    if repo_root is None:
        repo_root = clone_repo_if_needed()
        repo_root = find_repo_root()

    if repo_root is None:
        raise FileNotFoundError(
            'Could not find the project repository. Please ensure the repo is available at /content/project or in Drive.'
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

    # optional: extract audio using ffmpeg and pass audio_path into pipeline
    if extract_audio:
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
            extra_context["audio_path"] = str(audio_out.resolve())
        except Exception as exc:
            # continue but record failure in context
            extra_context["audio_extraction_error"] = str(exc)

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
    def extract_audio_only(video_file: str | Path | None) -> tuple[str, str, str, str]:
        try:
            if video_file is None:
                return 'no-input', '', 'Please upload a video file.', ''
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
                return 'audio_extracted', '', f'Audio extracted: {audio_out}', str(audio_out)
            except Exception as exc:
                return 'error', '', f'Audio extraction failed: {exc}', ''
        except Exception as exc:
            return 'error', '', str(exc), ''

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
            yield 'No video uploaded', '', '', ''
            return

        yield 'Starting pipeline...', '', '', ''

        audio_path = ''

        events: list[str] = []
        log_lines: list[str] = []

        def prettify(msg: str) -> str:
            try:
                if msg.startswith("start:"):
                    step = msg.split(":", 1)[1]
                    return f"Starting step: {step}"
                if msg.startswith("end:"):
                    step = msg.split(":", 1)[1]
                    return f"Finished step: {step}"
                if msg.startswith("download:"):
                    return f"Download status: {msg.split(':',1)[1]}"
                if msg.startswith("asr:"):
                    text = msg.split(":", 1)[1]
                    return f"ASR result: {text[:300]}"
                if msg.startswith("translate:"):
                    text = msg.split(":", 1)[1]
                    return f"Translation: {text[:300]}"
                if msg.startswith("render:"):
                    status = msg.split(":", 1)[1]
                    return f"Render status: {status}"
                if msg.startswith("ffmpeg-extract:"):
                    return f"ffmpeg (extract): {msg.split(':',1)[1]}"
                if msg.startswith("ffmpeg:"):
                    return f"ffmpeg: {msg.split(':',1)[1]}"
            except Exception:
                pass
            return msg

        def progress_cb(message: str) -> None:
            # append raw message to events; prettify when consuming
            events.append(message)

        result_container: dict = {}

        def target() -> None:
            try:
                status, output_path, message = run_video_pipeline(
                    str(video_file),
                    extract_audio=False,
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

        # stream events while thread runs; accumulate prettified log
        while thread.is_alive() or events:
            while events:
                ev = events.pop(0)
                pretty = prettify(ev)
                log_lines.append(pretty)
                # yield accumulated log into the status textbox
                yield "\n".join(log_lines), '', '', audio_path
            time.sleep(0.2)

        # final result
        status, output_path, message = result_container.get('result', ('error', '', 'No result'))
        yield status, output_path, message, audio_path

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
            status_out = gr.Textbox(label='Progress log', interactive=False, lines=12)
        with gr.Row():
            rendered_out = gr.Textbox(label='Rendered output path', interactive=False)
        with gr.Row():
            info_out = gr.Textbox(label='Info / Drive copy status', interactive=False)
        with gr.Row():
            audio_out = gr.Textbox(label='Audio path (if extracted)', interactive=False)

        run_button.click(
            fn=process_stream,
            inputs=[video_input, extract_chk, subs_chk, subtitle_mode, translation_target, asr_lang],
            outputs=[status_out, rendered_out, info_out, audio_out],
        )

        extract_button.click(
            fn=extract_audio_only,
            inputs=[video_input],
            outputs=[status_out, rendered_out, info_out, audio_out],
        )

    demo.launch(share=True)


if __name__ == '__main__':
    install_dependencies()
    mount_drive()
    build_interface()
