from __future__ import annotations

import os
import shutil
import subprocess
import sys
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
    public_url = demo.queue().launch(share=True, debug=True)
    print("[INFO] Public Gradio Link:")
    print(public_url)


if __name__ == "__main__":
    main()
