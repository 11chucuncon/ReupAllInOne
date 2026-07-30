from __future__ import annotations

import json
import logging
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import cv2
import torch
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
        if self.propainter_dir.exists() and (self.propainter_dir / "inference_propainter.py").exists():
            return self.propainter_dir

        logger.info("ProPainter repository not found at %s; cloning it automatically", self.propainter_dir)
        self.propainter_dir.parent.mkdir(parents=True, exist_ok=True)
        self._run_command(["git", "clone", "https://github.com/sczhou/ProPainter.git", str(self.propainter_dir)])
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

    def _read_frames_with_opencv(self, video_path: str) -> tuple[torch.Tensor, float, tuple[int, int], str]:
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video for ProPainter fallback: {video_path}")

        frames: list[torch.Tensor] = []
        fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        while True:
            ret, frame = capture.read()
            if not ret:
                break
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(rgb_frame))

        capture.release()
        if not frames:
            raise RuntimeError(f"No frames were decoded from {video_path}")

        return torch.stack(frames), float(fps), (width, height), str(video_path)

    def _build_propaint_runner(self, input_video_path: str, mask_video_path: str, output_path: str) -> str:
        script_path = self.propainter_dir / "inference_propainter.py"
        args = [
            str(script_path),
            "-i",
            input_video_path,
            "-m",
            mask_video_path,
            "-o",
            str(output_path),
            "--mask_dilation",
            str(self.inpaint_config.get("mask_dilation", 8)),
            "--subvideo_length",
            str(self.inpaint_config.get("subvideo_length", 80)),
            "--raft_iter",
            str(self.inpaint_config.get("raft_iter", 20)),
        ]
        if self.inpaint_config.get("fp16", False):
            args.append("--fp16")

        wrapper_code = textwrap.dedent(
            f"""
            import json
            import runpy
            import sys

            import cv2
            import torch
            import torchvision.io

            def custom_read_video(filename, *args, **kwargs):
                video_path = filename if isinstance(filename, str) else kwargs.get('filename', '')
                capture = cv2.VideoCapture(str(video_path))
                if not capture.isOpened():
                    raise RuntimeError(f'Could not open video: {{video_path}}')

                fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frames = []
                while True:
                    ret, frame = capture.read()
                    if not ret:
                        break
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(torch.from_numpy(rgb_frame))

                capture.release()
                if not frames:
                    raise RuntimeError(f'No frames were decoded from {{video_path}}')

                vframes = torch.stack(frames)
                aframes = torch.empty((0, 0))
                info = {{'video_fps': fps}}
                return vframes, aframes, info

            def read_frame_from_videos(video_path, *args, **kwargs):
                capture = cv2.VideoCapture(str(video_path))
                if not capture.isOpened():
                    raise RuntimeError(f'Could not open video for ProPainter fallback: {{video_path}}')

                fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
                width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frames = []
                while True:
                    ret, frame = capture.read()
                    if not ret:
                        break
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(torch.from_numpy(rgb_frame))

                capture.release()
                if not frames:
                    raise RuntimeError(f'No frames were decoded from {{video_path}}')

                vframes = torch.stack(frames)
                return vframes, fps, (height, width), str(video_path)

            torchvision.io.read_video = custom_read_video
            sys.argv = {json.dumps(args)}
            runpy.run_path({json.dumps(str(script_path))}, run_name='__main__')
            """
        )
        return wrapper_code

    def _run_propaint_inference(self, input_video_path: str, mask_video_path: str, output_path: str) -> None:
        wrapper_code = self._build_propaint_runner(input_video_path, mask_video_path, str(output_path))
        self._run_command([sys.executable, "-c", wrapper_code], cwd=self.propainter_dir)

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

    def _build_mask_video(self, mask_frames_dir: str, output_mask_video_path: str) -> str:
        mask_dir = Path(mask_frames_dir).expanduser().resolve()
        output_path = Path(output_mask_video_path).expanduser().resolve()
        if not mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

        mask_files = sorted(mask_dir.glob("mask_*.png"))
        if not mask_files:
            raise RuntimeError(f"No mask frames were generated in {mask_dir}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        first_frame = cv2.imread(str(mask_files[0]), cv2.IMREAD_GRAYSCALE)
        if first_frame is None:
            raise RuntimeError(f"Could not read mask frame: {mask_files[0]}")

        height, width = first_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, 24.0, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not create mask video writer for {output_path}")

        try:
            for mask_file in mask_files:
                frame = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                if frame is None:
                    continue
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                writer.write(frame_bgr)
        finally:
            writer.release()

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
            mask_output_dir_str = self.cleaner.generate_dynamic_mask_sequence(input_video_path, str(mask_output_dir))
            mask_video_path = self._build_mask_video(mask_output_dir_str, str(output_path.parent / "propainter_masks.mp4"))

        self._run_propaint_inference(input_video_path, mask_video_path, str(output_path))
        return str(output_path)
