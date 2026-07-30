from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.request import urlretrieve

import torch


def resolve_project_root(start_path: Path) -> Path:
    """Locate the real project root by walking upward until the app entrypoints are found."""
    current_path = start_path.expanduser().resolve()
    for candidate in [current_path, *current_path.parents]:
        if (candidate / "app_gradio.py").exists() and (candidate / "pipeline.py").exists() and (candidate / "config").exists():
            return candidate
    return current_path


ROOT = resolve_project_root(Path(__file__).resolve().parent)
TEMP_DIR = ROOT / "temp"
OUTPUT_DIR = ROOT / "outputs"


def ensure_directories() -> None:
    """Create required directories if they do not already exist."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale_item in TEMP_DIR.iterdir():
        if stale_item.is_dir():
            shutil.rmtree(stale_item, ignore_errors=True)
        else:
            stale_item.unlink(missing_ok=True)


def ensure_propaint_repository(root: Path) -> Path:
    """Ensure the ProPainter repository and required weights exist before launch."""
    propainter_dir = root / "core" / "ProPainter"
    if (propainter_dir / "inference_propainter.py").exists() and (propainter_dir / "weights").exists():
        print("[INFO] ProPainter repository already available.")
        return propainter_dir

    print("[INFO] ProPainter repository not found. Cloning it automatically...")
    propainter_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "https://github.com/sczhou/ProPainter.git", str(propainter_dir)], check=True, cwd=str(root))

    weights_dir = propainter_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    required_weights = {
        "ProPainter.pth": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth",
        "recurrent_flow_completion.pth": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth",
        "raft-things.pth": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth",
    }
    for filename, url in required_weights.items():
        target_path = weights_dir / filename
        if not target_path.exists():
            print(f"[INFO] Downloading {filename}...")
            urlretrieve(url, target_path)
    return propainter_dir


def validate_launch_prerequisites(root: Path) -> None:
    propainter_dir = root / "core" / "ProPainter"
    required_weights = [
        propainter_dir / "weights" / "ProPainter.pth",
        propainter_dir / "weights" / "recurrent_flow_completion.pth",
        propainter_dir / "weights" / "raft-things.pth",
    ]
    missing_weights = [str(path) for path in required_weights if not path.exists()]
    if missing_weights:
        raise RuntimeError(f"Missing ProPainter weights: {missing_weights}")

    env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not env_key:
        print("[WARN] GEMINI/OPENROUTER API key is not set in the environment; the app may fail during rewrite steps.")


def try_gradio_share(demo, port: int = 7860) -> str | None:
    """Try to launch Gradio with a public gradio.live share link for up to 10 seconds."""
    print("[INFO] Attempting Gradio share launch...")

    def launch_share() -> None:
        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=True,
            debug=True,
            prevent_thread_lock=True,
            show_error=True,
            allowed_paths=[
                "/content/workspace",
                "/content/workspace/output",
                "/content/workspace/temp",
            ],
        )

    thread = threading.Thread(target=launch_share, daemon=True)
    thread.start()

    deadline = time.time() + 10
    while time.time() < deadline:
        share_url = getattr(demo, "share_url", None)
        if share_url:
            print("[INFO] Gradio share link created successfully.")
            print("=" * 55)
            print(f"PUBLIC LINK (GRADIO): {share_url}")
            print("=" * 55)
            return share_url
        time.sleep(0.5)

    try:
        demo.close()
    except Exception:
        pass

    return None


def start_ngrok_tunnel(port: int = 7860) -> str | None:
    """Fallback to pyngrok when gradio.live sharing is unavailable or drops."""
    try:
        from pyngrok import ngrok
    except ModuleNotFoundError:
        print("[INFO] Installing pyngrok for tunnel fallback...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyngrok"], check=False)
        from pyngrok import ngrok

    auth_token = os.environ.get("NGROK_AUTHTOKEN")
    if auth_token:
        ngrok.set_auth_token(auth_token)

    try:
        tunnel = ngrok.connect(port, "http")
        public_url = getattr(tunnel, "public_url", None) or str(tunnel)
        print("=" * 55)
        print(f"PUBLIC LINK (NGROK): {public_url}")
        print("=" * 55)
        return public_url
    except Exception as exc:
        print(f"[WARN] Ngrok tunnel could not be started: {exc}")
        return None


def main() -> None:
    """Launch the Gradio app in a Colab-friendly environment."""
    ensure_directories()
    ensure_propaint_repository(ROOT)
    validate_launch_prerequisites(ROOT)

    print("[INFO] Checking GPU availability...")
    if torch.cuda.is_available():
        print(f"[INFO] CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARN] CUDA GPU is not available. The app may run on CPU only.")

    print("[INFO] Installing Colab-compatible media dependencies for ProPainter fallback...")
    subprocess.run([sys.executable, "-m", "pip", "install", "av", "imageio[ffmpeg]"], check=False)

    sys.path.insert(0, str(ROOT))

    try:
        import app_gradio
    except ModuleNotFoundError as exc:
        print(f"[ERROR] Could not import app_gradio.py: {exc}")
        raise

    print("[INFO] Starting Gradio app on Colab...")
    demo = app_gradio.create_app()
    demo.queue(default_concurrency_limit=1)

    share_url = try_gradio_share(demo, port=7860)
    if share_url:
        print("[INFO] Gradio is running with the public share link above.")
        while True:
            time.sleep(60)
        return

    print("[INFO] Gradio share link was not available; falling back to Ngrok...")

    launch_thread = threading.Thread(
        target=lambda: demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True,
            prevent_thread_lock=True,
            show_error=True,
            allowed_paths=[
                "/content/workspace",
                "/content/workspace/output",
                "/content/workspace/temp",
            ],
        ),
        daemon=True,
    )
    launch_thread.start()
    time.sleep(2)

    start_ngrok_tunnel(port=7860)

    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
