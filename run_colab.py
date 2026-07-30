from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlretrieve

import torch


ROOT = Path(__file__).resolve().parent
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


def start_localtunnel(port: int = 7860, subdomain: str = "reupstudio") -> subprocess.Popen[str] | None:
    """Attempt to start a fixed localtunnel route for the Gradio server."""
    try:
        command = ["npx", "--yes", "localtunnel", "--port", str(port), "--subdomain", subdomain]
        print(f"[INFO] Starting localtunnel with: {' '.join(command)}")
        return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except FileNotFoundError:
        print("[WARN] localtunnel could not be started because 'npx' is not available. Install Node.js/npm and run the command manually.")
        return None
    except Exception as exc:
        print(f"[WARN] localtunnel launch failed: {exc}")
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
    tunnel_process = start_localtunnel(port=7860)
    if tunnel_process is not None:
        time.sleep(3)
        print("[INFO] localtunnel started in the background. If it fails, run this manually:")
        print("[INFO] npx --yes localtunnel --port 7860 --subdomain reupstudio")

    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
    )
    if tunnel_process is not None:
        tunnel_process.terminate()


if __name__ == "__main__":
    main()
