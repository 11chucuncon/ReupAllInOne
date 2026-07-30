from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import cv2
import torch
import yaml

from config import CLEANED_DIR, CLEANED_VIDEO_PATH, TEMP_DIR, WORK_DIR, find_file_anywhere, promote_file_to_destination
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
        self.workspace_temp_dir = TEMP_DIR
        self.workspace_cleaned_dir = CLEANED_DIR
        self.cleaned_video_path = CLEANED_VIDEO_PATH

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
        self.propainter_dir = (self.project_root / "core" / "ProPainter").resolve()
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

    def _resize_video_for_inpainting(self, input_video_path: str, output_path: str, resize_max_side: int = 1280) -> str:
        input_path = Path(input_video_path).expanduser().resolve()
        output_file = Path(output_path).expanduser().resolve()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video for resize preprocessing: {input_video_path}")

        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width <= 0 or height <= 0:
                return input_video_path
            if max(width, height) <= resize_max_side:
                return input_video_path

            if width >= height:
                new_width = resize_max_side
                new_height = int(height * resize_max_side / width)
            else:
                new_height = resize_max_side
                new_width = int(width * resize_max_side / height)

            writer = None
            while True:
                ret, frame = capture.read()
                if not ret:
                    break
                resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(output_file), fourcc, max(12.0, capture.get(cv2.CAP_PROP_FPS) or 24.0), (new_width, new_height))
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not create resized video writer: {output_file}")
                writer.write(resized)
            if writer is not None:
                writer.release()
            capture.release()
            return str(output_file)
        finally:
            capture.release()

    def _get_propaint_resize_args(self, input_video_path: str, resize_max_side: int = 1280) -> list[str]:
        capture = cv2.VideoCapture(input_video_path)
        if not capture.isOpened():
            logger.warning("Could not inspect video dimensions for ProPainter resize; skipping resize cap")
            return []

        try:
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            capture.release()

        if width <= 0 or height <= 0:
            return []

        max_side = max(width, height)
        if max_side <= resize_max_side:
            return []

        if width >= height:
            return ["--width", str(resize_max_side)]
        return ["--height", str(resize_max_side)]

    def _build_propaint_runner(
        self,
        input_video_path: str,
        mask_video_path: str,
        output_path: str,
        subvideo_length: int = 30,
        raft_iter: int = 10,
        resize_max_side: int = 1280,
        fp16: bool = True,
        enable_vram_cleanup: bool = True,
    ) -> str:
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
            str(subvideo_length or self.inpaint_config.get("subvideo_length", 30)),
            "--raft_iter",
            str(raft_iter or self.inpaint_config.get("raft_iter", 10)),
        ]
        args.extend(self._get_propaint_resize_args(input_video_path, resize_max_side=resize_max_side))
        if fp16 or self.inpaint_config.get("fp16", False):
            args.append("--fp16")

        wrapper_code = textwrap.dedent(
            f"""
            import json
            import runpy
            import sys

            import cv2
            import torch
            import torchvision.io

            import gc
            import torch

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
            if {repr(enable_vram_cleanup)}:
                gc.collect()
                torch.cuda.empty_cache()
            sys.argv = {json.dumps(args)}
            runpy.run_path({json.dumps(str(script_path))}, run_name='__main__')
            """
        )
        return wrapper_code

    def _run_propaint_inference(
        self,
        input_video_path: str,
        mask_video_path: str,
        output_path: str,
        subvideo_length: int = 30,
        raft_iter: int = 10,
        resize_max_side: int = 1280,
        fp16: bool = True,
        enable_vram_cleanup: bool = True,
    ) -> None:
        wrapper_code = self._build_propaint_runner(
            input_video_path,
            mask_video_path,
            str(output_path),
            subvideo_length=subvideo_length,
            raft_iter=raft_iter,
            resize_max_side=resize_max_side,
            fp16=fp16,
            enable_vram_cleanup=enable_vram_cleanup,
        )
        self._run_command([sys.executable, "-c", wrapper_code], cwd=self.propainter_dir)

    def _prepare_output_path(self, output_video_path: str) -> Path:
        output_path = Path(output_video_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            if output_path.is_dir():
                shutil.rmtree(output_path)
            else:
                os.remove(output_path)
        return output_path

    def _resolve_output_video_path(self, output_path: Path) -> Path:
        if output_path.exists() and output_path.is_file():
            return output_path

        if output_path.exists() and output_path.is_dir():
            try:
                candidate = find_file_anywhere(output_path, ".mp4")
            except FileNotFoundError:
                shutil.rmtree(output_path, ignore_errors=True)
                return output_path

            output_path.parent.mkdir(parents=True, exist_ok=True)
            return promote_file_to_destination(candidate, output_path, search_root=output_path, move=True)

        return output_path

    def _blur_video(self, input_video_path: str, output_video_path: str) -> str:
        output_path = self._prepare_output_path(output_video_path)
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

    def _prepare_mask_sequence_directory(self, mask_source: str, output_dir: str) -> str:
        source_path = Path(mask_source).expanduser().resolve()
        output_path = Path(output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        if source_path.is_dir():
            mask_files = sorted(source_path.glob("mask_*.png"))
            if not mask_files:
                raise RuntimeError(f"No mask frames were generated in {source_path}")
            for index, mask_file in enumerate(mask_files):
                target_file = output_path / f"{index:05d}.png"
                target_file.write_bytes(mask_file.read_bytes())
            return str(output_path)

        if source_path.suffix.lower() == ".mp4":
            capture = cv2.VideoCapture(str(source_path))
            if not capture.isOpened():
                raise RuntimeError(f"Could not open mask video: {source_path}")

            frame_index = 0
            while True:
                ret, frame = capture.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(str(output_path / f"{frame_index:05d}.png"), gray)
                frame_index += 1
            capture.release()
            if frame_index == 0:
                raise RuntimeError(f"No frames were extracted from mask video {source_path}")
            return str(output_path)

        raise RuntimeError(f"Unsupported mask source: {mask_source}")

    def clean_video(
        self,
        input_video_path: str,
        output_video_path: str,
        mode: str = "propainter",
        mask_video_path: Optional[str] = None,
        subvideo_length: int = 30,
        raft_iter: int = 10,
        resize_max_side: int = 1280,
        fp16: bool = True,
        enable_vram_cleanup: bool = True,
    ) -> str:
        output_path = self._prepare_output_path(output_video_path)
        self.workspace_cleaned_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_temp_dir.mkdir(parents=True, exist_ok=True)

        if mode == "blur":
            return self._blur_video(input_video_path, str(output_path))

        self._ensure_propaint_repository()
        resized_input_path = self._resize_video_for_inpainting(input_video_path, str(output_path.parent / "resized_input.mp4"), resize_max_side=resize_max_side)

        if mask_video_path is None:
            mask_output_dir = output_path.parent / "propainter_masks"
            mask_output_dir_str = self.cleaner.generate_dynamic_mask_sequence(input_video_path, str(mask_output_dir))
            mask_video_path = self._prepare_mask_sequence_directory(mask_output_dir_str, str(output_path.parent / "propainter_masks_dir"))
        else:
            mask_video_path = self._prepare_mask_sequence_directory(mask_video_path, str(output_path.parent / "propainter_masks_dir"))

        self._run_propaint_inference(
            input_video_path,
            mask_video_path,
            str(output_path),
            subvideo_length=subvideo_length,
            raft_iter=raft_iter,
            resize_max_side=resize_max_side,
            fp16=fp16,
            enable_vram_cleanup=enable_vram_cleanup,
        )

        cleaned_search_root = WORK_DIR / "cleaned"
        try:
            discovered_video = find_file_anywhere(cleaned_search_root, ".mp4")
        except FileNotFoundError:
            discovered_video = self._resolve_output_video_path(output_path)

        if discovered_video.exists() and discovered_video.is_file() and discovered_video != self.cleaned_video_path:
            discovered_video = promote_file_to_destination(discovered_video, self.cleaned_video_path, search_root=cleaned_search_root, move=True)

        resolved_output_path = discovered_video
        if not resolved_output_path.exists() or resolved_output_path.is_dir():
            raise RuntimeError(f"ProPainter did not produce a valid video file at {resolved_output_path}")
        return str(resolved_output_path)
