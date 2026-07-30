from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
TEMP_DIR = ROOT / "temp"
OUTPUT_DIR = ROOT / "outputs"


def ensure_directories() -> None:
    """Create required directories if they do not already exist."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Launch the Gradio app in a Colab-friendly environment."""
    ensure_directories()

    print("[INFO] Checking GPU availability...")
    if torch.cuda.is_available():
        print(f"[INFO] CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARN] CUDA GPU is not available. The app may run on CPU only.")

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
