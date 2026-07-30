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
from typing import Optional, Sequence
from urllib.request import urlretrieve

import cv2
import numpy as np
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
        try:
            completed = subprocess.run(command, check=True, cwd=str(cwd or self.project_root), capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            stderr_output = exc.stderr or ""
            stdout_output = exc.stdout or ""
            logger.error("Inpainter command failed with exit code %s", exc.returncode)
            if stdout_output:
                logger.error("Inpainter stdout:\n%s", stdout_output)
            if stderr_output:
                logger.error("Inpainter stderr:\n%s", stderr_output)
            print("[ERROR] ProPainter command failed")
            if stdout_output:
                print(stdout_output)
            if stderr_output:
                print(stderr_output)
            raise
        if completed.stdout:
            logger.info("Inpainter stdout:\n%s", completed.stdout)
        if completed.stderr:
            logger.info("Inpainter stderr:\n%s", completed.stderr)

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

        safe_resize = max(320, int(resize_max_side * 0.75))
        if width >= height:
            return ["--width", str(safe_resize)]
        return ["--height", str(safe_resize)]

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
        mask_path = Path(mask_video_path).expanduser().resolve()
        if not mask_path.exists():
            logger.warning("Mask file does not exist before ProPainter run: %s", mask_path)
            print(f"[WARNING] Mask file missing: {mask_path}")
            raise FileNotFoundError(f"Mask file does not exist: {mask_path}")

        if enable_vram_cleanup:
            try:
                import gc
                import torch as torch_module

                gc.collect()
                torch_module.cuda.empty_cache()
            except Exception as exc:
                logger.warning("VRAM cleanup failed: %s", exc)

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
        try:
            self._run_command([sys.executable, "-c", wrapper_code], cwd=self.propainter_dir)
        except Exception:
            if enable_vram_cleanup:
                try:
                    import gc
                    import torch as torch_module

                    gc.collect()
                    torch_module.cuda.empty_cache()
                except Exception:
                    pass
            raise

    def _has_valid_mask(self, mask_source: Optional[str]) -> bool:
        if not mask_source:
            return False

        mask_path = Path(mask_source).expanduser().resolve()
        if not mask_path.exists():
            return False

        if mask_path.is_dir():
            mask_files = [child for child in mask_path.iterdir() if child.is_file() and child.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}]
            return bool(mask_files)

        if mask_path.is_file():
            return mask_path.stat().st_size > 0

        return False

    def _boxes_from_mask(self, mask: np.ndarray) -> list[tuple[int, int, int, int]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[tuple[int, int, int, int]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w <= 8 or h <= 8:
                continue
            boxes.append((x, y, w, h))
        return boxes

    def detect_watermark_and_text(self, input_video_path: str) -> list[list[tuple[int, int, int, int]]]:
        input_path = Path(input_video_path).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_video_path}")

        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video for YOLO detection: {input_video_path}")

        boxes_by_frame: list[list[tuple[int, int, int, int]]] = []
        frame_index = 0

        while True:
            ret, frame = capture.read()
            if not ret:
                break

            frame_index += 1
            yolo_mask = self.cleaner._run_yolo(frame)
            text_mask = self.cleaner._extract_text_mask(frame)
            combined_mask = cv2.bitwise_or(yolo_mask, text_mask)
            if np.count_nonzero(combined_mask) == 0:
                continue

            if self.cleaner._create_motion_mask is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (9, 9), 0)
                mask_motion = self.cleaner._create_motion_mask(blur, blur)
                combined_mask = cv2.bitwise_or(combined_mask, mask_motion)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)

            frame_boxes = self._boxes_from_mask(combined_mask)
            if frame_boxes:
                boxes_by_frame.append(frame_boxes)

        capture.release()
        return boxes_by_frame

    def _build_mask_sequence_from_boxes(
        self,
        input_video_path: str,
        boxes_by_frame: list[list[tuple[int, int, int, int]]],
        output_dir: str,
    ) -> str:
        input_path = Path(input_video_path).expanduser().resolve()
        output_path = Path(output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video for mask generation: {input_video_path}")

        frame_index = 0
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            if frame_index < len(boxes_by_frame):
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                for x, y, w, h in boxes_by_frame[frame_index]:
                    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
            else:
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)

            mask_path = output_path / f"mask_{frame_index:05d}.png"
            cv2.imwrite(str(mask_path), mask)
            frame_index += 1

        capture.release()
        if frame_index == 0:
            raise RuntimeError(f"No frames were processed for mask generation from {input_video_path}")
        return str(output_path)

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

            if candidate and candidate.exists():
                output_path.parent.mkdir(parents=True, exist_ok=True)
                return promote_file_to_destination(candidate, output_path, search_root=output_path, move=True)

            shutil.rmtree(output_path, ignore_errors=True)
            return output_path

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
        detected_boxes: Optional[list[list[tuple[int, int, int, int]]]] = None,
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

        if not detected_boxes and not self._has_valid_mask(mask_video_path):
            logger.info("[INFO] No watermark mask detected. Skipping ProPainter step.")
            print("[INFO] No watermark mask detected. Skipping ProPainter step.")
            for target_path in {output_path, self.cleaned_video_path}:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(input_video_path), str(target_path))
            return str(output_path)

        self._ensure_propaint_repository()
        resized_input_path = self._resize_video_for_inpainting(input_video_path, str(output_path.parent / "resized_input.mp4"), resize_max_side=resize_max_side)

        if detected_boxes:
            mask_output_dir = output_path.parent / "propainter_masks"
            mask_output_dir_str = self._build_mask_sequence_from_boxes(input_video_path, detected_boxes, str(mask_output_dir))
            mask_video_path = self._prepare_mask_sequence_directory(mask_output_dir_str, str(output_path.parent / "propainter_masks_dir"))
        elif mask_video_path is None:
            mask_output_dir = output_path.parent / "propainter_masks"
            mask_output_dir_str = self.cleaner.generate_dynamic_mask_sequence(input_video_path, str(mask_output_dir))
            mask_video_path = self._prepare_mask_sequence_directory(mask_output_dir_str, str(output_path.parent / "propainter_masks_dir"))
        else:
            mask_video_path = self._prepare_mask_sequence_directory(mask_video_path, str(output_path.parent / "propainter_masks_dir"))

        try:
            if enable_vram_cleanup:
                try:
                    import gc
                    import torch as torch_module

                    gc.collect()
                    torch_module.cuda.empty_cache()
                except Exception as exc:
                    logger.warning("VRAM cleanup before ProPainter failed: %s", exc)

            # Enforce smaller inference resolution and shorter segments to reduce VRAM spikes.
            safe_resize_max_side = min(resize_max_side, 848)
            safe_subvideo_length = min(subvideo_length, 20)
            self._run_propaint_inference(
                resized_input_path,
                mask_video_path,
                str(output_path),
                subvideo_length=safe_subvideo_length,
                raft_iter=raft_iter,
                resize_max_side=safe_resize_max_side,
                fp16=fp16,
                enable_vram_cleanup=enable_vram_cleanup,
            )
        except Exception as exc:
            logger.warning("[WARNING] ProPainter inpainting failed (%s). Falling back to original input video.", exc)
            print(f"[WARNING] ProPainter inpainting failed ({exc}). Falling back to original input video.")
            if Path(input_video_path).exists():
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(input_video_path), str(output_path))
                return str(output_path)
            raise RuntimeError(f"ProPainter failed and source video is missing: {input_video_path}") from exc

        if output_path.exists() and output_path.is_file():
            return str(output_path)

        if output_path.exists() and output_path.is_dir():
            try:
                candidate = find_file_anywhere(output_path, ".mp4")
            except FileNotFoundError:
                shutil.rmtree(output_path, ignore_errors=True)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(input_video_path), str(output_path))
                return str(output_path)

            if candidate and candidate.exists():
                return str(promote_file_to_destination(candidate, output_path, search_root=output_path, move=True))

            shutil.rmtree(output_path, ignore_errors=True)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(input_video_path), str(output_path))
            return str(output_path)

        if Path(input_video_path).exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(input_video_path), str(output_path))
            return str(output_path)

        raise RuntimeError(f"Unable to resolve ProPainter output or fallback from {input_video_path}")
