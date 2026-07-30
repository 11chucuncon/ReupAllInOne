from __future__ import annotations

import logging
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import yaml

from core.cleaner import VideoCleaner

logger = logging.getLogger(__name__)


class VideoInpainter:
    """Run ProPainter inpainting on a video using masks to remove subtitles and watermarks."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.config_path = config_path or str(self.project_root / "config" / "settings.yaml")
        self.settings = self._load_settings()
        self.inpaint_config = self.settings.get("inpaint", {})
        self.propainter_dir = self.project_root / "core" / "ProPainter"
        self.cleaner = VideoCleaner(config_path=self.config_path)

    def _load_settings(self) -> dict:
        try:
            with open(self.config_path, "r", encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
        except FileNotFoundError as exc:
            logger.error("Settings file not found at %s", self.config_path)
            raise
        except yaml.YAMLError as exc:
            logger.error("Invalid YAML configuration: %s", exc)
            raise

    def _run_command(self, command: list[str], cwd: Optional[Path] = None) -> None:
        logger.info("Running inpainter command: %s", " ".join(shlex.quote(part) for part in command))
        subprocess.run(command, check=True, cwd=str(cwd or self.project_root))

    def _ensure_propaint_repository(self) -> Path:
        if self.propainter_dir.exists() and (self.propainter_dir / "run_video.py").exists():
            return self.propainter_dir

        logger.info("ProPainter repository not found at %s; cloning it automatically", self.propainter_dir)
        self.propainter_dir.parent.mkdir(parents=True, exist_ok=True)
        self._run_command(["git", "clone", "https://github.com/sczhou/ProPainter.git", str(self.propainter_dir)])

        run_video_script = self.propainter_dir / "run_video.py"
        if not run_video_script.exists():
            run_video_script.write_text(
                """#!/usr/bin/env python3\nfrom __future__ import annotations\n\nimport argparse\nimport subprocess\nimport sys\nfrom pathlib import Path\n\nROOT = Path(__file__).resolve().parent\n\ndef main() -> None:\n    parser = argparse.ArgumentParser(description='High-quality ProPainter video inpainting wrapper')\n    parser.add_argument('--input', required=True)\n    parser.add_argument('--output', required=True)\n    parser.add_argument('--mask', required=True)\n    parser.add_argument('--model_path', default=str(ROOT / 'weights' / 'ProPainter.pth'))\n    parser.add_argument('--mask_dilation', type=int, default=8)\n    parser.add_argument('--subvideo_length', type=int, default=80)\n    parser.add_argument('--raft_iter', type=int, default=20)\n    parser.add_argument('--fp16', action='store_true')\n    args = parser.parse_args()\n\n    cmd = [sys.executable, str(ROOT / 'inference_propainter.py'), '-i', args.input, '-m', args.mask, '-o', args.output, '--mask_dilation', str(args.mask_dilation), '--subvideo_length', str(args.subvideo_length), '--raft_iter', str(args.raft_iter)]\n    if args.fp16:\n        cmd.append('--fp16')\n    subprocess.run(cmd, check=True, cwd=str(ROOT))\n\n\nif __name__ == '__main__':\n    main()\n""",
                encoding="utf-8",
            )

        self._ensure_propaint_weights()
        return self.propainter_dir

    def _ensure_propaint_weights(self) -> None:
        weights_dir = self.propainter_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        required_weights = {
            "ProPainter.pth": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth",
            "recurrent_flow_completion.pth": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth",
            "raft-things.pth": "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth",
        }
        for filename, url in required_weights.items():
            target_path = weights_dir / filename
            if target_path.exists():
                continue
            logger.info("Downloading ProPainter weight: %s", filename)
            urlretrieve(url, target_path)

    def _blur_video(self, input_video_path: str, output_video_path: str) -> str:
        output_path = Path(output_video_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-y",
            "-i",
            input_video_path,
            "-vf",
            "boxblur=10:1",
            "-c:a",
            "copy",
            str(output_path),
        ]
        self._run_command(command)
        return str(output_path)

    def clean_video(
        self,
        input_video_path: str,
        output_video_path: str,
        mode: str = "propainter",
        mask_video_path: Optional[str] = None,
    ) -> str:
        output_path = Path(output_video_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if mode == "blur":
            return self._blur_video(input_video_path, str(output_path))

        self._ensure_propaint_repository()

        if mask_video_path is None:
            mask_output_dir = output_path.parent / "propainter_masks"
            mask_video_path = self.cleaner.generate_dynamic_mask_sequence(input_video_path, str(mask_output_dir))

        command = [
            sys.executable,
            str(self.propainter_dir / "run_video.py"),
            "--input",
            input_video_path,
            "--output",
            str(output_path),
            "--model_path",
            str(self.propainter_dir / "weights" / "ProPainter.pth"),
        ]
        if mask_video_path:
            command.extend(["--mask", mask_video_path])

        self._run_command(command)
        return str(output_path)
