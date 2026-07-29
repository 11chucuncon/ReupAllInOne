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


def run_video_pipeline(video_path: str) -> tuple[str, str, str]:
    repo_root = setup_repository()
    from app.plugins.runner import run_from_config

    input_path = Path(video_path)
    if not input_path.exists():
        raise FileNotFoundError(f'Video file not found: {video_path}')

    result = run_from_config('config_pipeline_full.yaml', video_url=str(input_path.resolve()))
    rendered_path = Path(result.get('rendered_path', 'outputs/final.mp4'))
    if not rendered_path.is_absolute():
        rendered_path = (repo_root / rendered_path).resolve()

    if not rendered_path.exists():
        raise FileNotFoundError(f'Rendered output not found: {rendered_path}')

    output_folder = Path('/content/drive/MyDrive/colab_outputs')
    if output_folder.exists() or output_folder.parent.exists():
        try:
            output_folder.mkdir(parents=True, exist_ok=True)
            saved_path = output_folder / rendered_path.name
            saved_path.write_bytes(rendered_path.read_bytes())
            output_message = f'Output copied to Drive: {saved_path}'
        except Exception as exc:
            output_message = f'Output saved locally at {rendered_path}, but could not copy to Drive: {exc}'
    else:
        output_message = f'Output saved locally at {rendered_path}'
    return result.get('render_status', 'unknown'), str(rendered_path), output_message


def build_interface() -> None:
    import gradio as gr

    def process(video_file: Path | None) -> tuple[str, str, str]:
        if video_file is None:
            return 'No video uploaded', '', ''
        video_path = str(video_file)
        try:
            status, output_path, message = run_video_pipeline(video_path)
            return status, output_path, message
        except Exception as exc:
            return 'error', '', str(exc)

    with gr.Blocks() as demo:
        gr.Markdown('# Video Pipeline Colab UI')
        gr.Markdown(
            'Upload a video and click **Run**. The pipeline will execute on Colab and return the rendered video path.'
        )
        with gr.Row():
            video_input = gr.File(label='Upload video file', file_count='single', type='filepath')
        with gr.Row():
            run_button = gr.Button('Run pipeline')
        status_output = gr.Textbox(label='Render status', interactive=False)
        output_path = gr.Textbox(label='Rendered output path', interactive=False)
        message_output = gr.Textbox(label='Info / Drive copy status', interactive=False)

        run_button.click(process, inputs=[video_input], outputs=[status_output, output_path, message_output])

    demo.launch(share=True)


if __name__ == '__main__':
    install_dependencies()
    mount_drive()
    build_interface()
