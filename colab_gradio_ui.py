from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


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


def find_repo_root() -> Path | None:
    candidates = [
        Path.cwd(),
        Path('/content/project'),
        Path('/content/video-pipeline'),
        Path('/content/drive/MyDrive/A'),
        Path('/content/drive/MyDrive/video-pipeline'),
        Path('/content/drive/My Drive/A'),
        Path('/content/drive/My Drive/video-pipeline'),
    ]
    for candidate in candidates:
        if (candidate / 'app' / 'plugins' / 'runner.py').exists():
            return candidate
    return None


def clone_repo_if_needed() -> Path:
    repo_root = Path('/content/project')
    if (repo_root / 'app' / 'plugins' / 'runner.py').exists():
        return repo_root

    if repo_root.exists():
        shutil.rmtree(repo_root, ignore_errors=True)

    for url in ['https://github.com/hanhwannau/A.git', 'https://github.com/hanhwannau/A']:
        for branch in ['master', 'main']:
            print(f'Trying {url} branch {branch}...')
            try:
                subprocess.run(
                    ['git', 'clone', '--depth', '1', '--branch', branch, url, str(repo_root)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                print('Clone succeeded')
                return repo_root
            except subprocess.CalledProcessError as exc:
                print('Clone failed:', exc.returncode)
                print('stdout:', exc.stdout)
                print('stderr:', exc.stderr)

    raise RuntimeError('Failed to clone repository from GitHub. Check connectivity/authentication and whether the repo is public.')


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

    result = run_from_config("config_pipeline_full.yaml", video_url=str(input_path.resolve()), extra_context=extra_context)
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

    def process(
        video_file: str | Path | None,
        extract_audio: bool,
        generate_subtitles: bool,
        subtitle_mode: str,
        translation_target: str,
        asr_language: str,
    ) -> tuple[str, str, str]:
        try:
            if video_file is None:
                return 'No video uploaded', '', 'Please upload a video file.'

            video_path = str(video_file)
            status, output_path, message = run_video_pipeline(
                video_path,
                extract_audio=extract_audio,
                generate_subtitles=generate_subtitles,
                subtitle_mode=subtitle_mode,
                translation_target=translation_target or None,
                asr_language=asr_language or None,
            )
            return status, output_path, message
        except Exception as exc:
            return 'error', '', str(exc)

    inputs = [
        gr.File(label='Upload video file', file_count='single', type='filepath'),
        gr.Checkbox(label='Extract audio (wav)', value=False),
        gr.Checkbox(label='Generate subtitles', value=True),
        gr.Radio(['translated', 'source'], label='Subtitle mode', value='translated'),
        gr.Dropdown(['auto','en','vi','zh','fr','es'], label='Translation target language', value='vi'),
        gr.Textbox(label='ASR language (e.g. en, vi) or "auto"', value='auto'),
    ]
    outputs = [
        gr.Textbox(label='Render status', interactive=False),
        gr.Textbox(label='Rendered output path', interactive=False),
        gr.Textbox(label='Info / Drive copy status', interactive=False),
    ]

    iface = gr.Interface(
        fn=process,
        inputs=inputs,
        outputs=outputs,
        title='Video Pipeline Colab UI',
        description='Upload a video and click Run. The pipeline will execute on Colab and return the rendered video path.',
    )

    iface.launch(share=True)


if __name__ == '__main__':
    install_dependencies()
    mount_drive()
    build_interface()
