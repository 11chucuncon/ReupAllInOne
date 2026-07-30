from __future__ import annotations

import os
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


def ensure_propaint_repository(root: Path) -> Path:
    """Ensure the ProPainter repository and wrapper entrypoint exist before launch."""
    propainter_dir = root / "core" / "ProPainter"
    run_video_script = propainter_dir / "run_video.py"
    inference_script = propainter_dir / "inference_propainter.py"

    if run_video_script.exists() and inference_script.exists():
        print("[INFO] ProPainter repository already available.")
        return propainter_dir

    print("[INFO] ProPainter repository not found. Cloning it automatically...")
    propainter_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "https://github.com/sczhou/ProPainter.git", str(propainter_dir)], check=True, cwd=str(root))

    if not run_video_script.exists():
        run_video_script.write_text(
            """#!/usr/bin/env python3\nfrom __future__ import annotations\n\nimport argparse\nimport subprocess\nimport sys\nfrom pathlib import Path\n\nROOT = Path(__file__).resolve().parent\n\ndef main() -> None:\n    parser = argparse.ArgumentParser(description='High-quality ProPainter video inpainting wrapper')\n    parser.add_argument('--input', required=True)\n    parser.add_argument('--output', required=True)\n    parser.add_argument('--mask', required=True)\n    parser.add_argument('--model_path', default=str(ROOT / 'weights' / 'ProPainter.pth'))\n    parser.add_argument('--mask_dilation', type=int, default=8)\n    parser.add_argument('--subvideo_length', type=int, default=80)\n    parser.add_argument('--raft_iter', type=int, default=20)\n    parser.add_argument('--fp16', action='store_true')\n    args = parser.parse_args()\n\n    cmd = [\n        sys.executable,\n        str(ROOT / 'inference_propainter.py'),\n        '-i',\n        args.input,\n        '-m',\n        args.mask,\n        '-o',\n        args.output,\n        '--mask_dilation',\n        str(args.mask_dilation),\n        '--subvideo_length',\n        str(args.subvideo_length),\n        '--raft_iter',\n        str(args.raft_iter),\n    ]\n    if args.fp16:\n        cmd.append('--fp16')\n    subprocess.run(cmd, check=True, cwd=str(ROOT))\n\n\nif __name__ == '__main__':\n    main()\n""",
            encoding="utf-8",
        )

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


def main() -> None:
    """Launch the Gradio app in a Colab-friendly environment."""
    ensure_directories()
    ensure_propaint_repository(ROOT)

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
