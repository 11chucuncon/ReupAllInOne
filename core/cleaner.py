from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import yaml

from core.detector import YOLOv8SegmentationDetector

logger = logging.getLogger(__name__)


def _load_settings(config_path: str | None = None) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    config_path = config_path or str(project_root / "config" / "settings.yaml")
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except FileNotFoundError as exc:
        logger.error("Settings file not found at %s", config_path)
        raise
    except yaml.YAMLError as exc:
        logger.error("Invalid YAML configuration: %s", exc)
        raise


class VideoCleaner:
    """Remove moving watermarks and logos using motion detection and inpainting."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.settings = _load_settings(config_path)
        self.cleaner_config = self.settings.get("cleaner", {})
        self.min_area = int(self.cleaner_config.get("min_logo_area", 400))
        self.inpaint_radius = int(self.cleaner_config.get("inpaint_radius", 3))
        self.inpaint_method = self.cleaner_config.get("inpaint_method", "telea").lower()
        self.sample_rate = int(self.cleaner_config.get("sample_rate", 1))
        self.yolo_confidence = float(self.cleaner_config.get("yolo_confidence", 0.35))
        self.yolo_nms_threshold = float(self.cleaner_config.get("yolo_nms_threshold", 0.45))
        self.detector = YOLOv8SegmentationDetector(config_path=config_path)

    def _get_inpaint_flag(self) -> int:
        if self.inpaint_method == "ns":
            return cv2.INPAINT_NS
        return cv2.INPAINT_TELEA

    def _create_motion_mask(self, prev_gray: np.ndarray, current_gray: np.ndarray) -> np.ndarray:
        delta = cv2.absdiff(prev_gray, current_gray)
        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
        return opened

    def _extract_logo_region(self, mask: np.ndarray) -> np.ndarray:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logo_mask = np.zeros_like(mask)
        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(logo_mask, (x, y), (x + w, y + h), 255, -1)
        return logo_mask

    def _extract_text_mask(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        gradient = cv2.morphologyEx(
            blur,
            cv2.MORPH_GRADIENT,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )
        _, thresh = cv2.threshold(gradient, 20, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
        dilated = cv2.dilate(opened, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5)), iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fallback_mask = np.zeros_like(dilated)
        height, width = fallback_mask.shape[:2]
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < 30 or h < 10 or h > height * 0.3:
                continue
            cv2.rectangle(fallback_mask, (x, y), (x + w, y + h), 255, -1)
        return fallback_mask

    def _resolve_model_path(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return Path(__file__).resolve().parents[1] / path

    def _run_yolo(self, frame: np.ndarray) -> np.ndarray:
        return self.detector.detect_mask(frame)

    def generate_dynamic_mask_sequence(self, input_video_path: str, output_mask_dir: str) -> str:
        input_path = Path(input_video_path).expanduser().resolve()
        output_dir = Path(output_mask_dir).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_video_path}")
        output_dir.mkdir(parents=True, exist_ok=True)

        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video file: {input_video_path}")

        prev_gray: Optional[np.ndarray] = None
        frame_index = 0
        written_masks = 0

        while True:
            ret, frame = capture.read()
            if not ret:
                break

            frame_index += 1
            if self.sample_rate > 1 and frame_index % self.sample_rate != 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (9, 9), 0)

            text_mask = self._extract_text_mask(frame)
            yolo_mask = self._run_yolo(frame)
            frame_mask = cv2.bitwise_or(text_mask, yolo_mask)

            if prev_gray is not None:
                motion_mask = self._create_motion_mask(prev_gray, blur)
                logo_mask = self._extract_logo_region(motion_mask)
                frame_mask = cv2.bitwise_or(frame_mask, logo_mask)

            prev_gray = blur

            if np.count_nonzero(frame_mask) == 0:
                height, width = frame.shape[:2]
                pad = int(min(width, height) * 0.02)
                bottom_strip = int(height * 0.12)
                cv2.rectangle(frame_mask, (pad, height - bottom_strip), (width - pad, height - pad), 255, -1)
                cv2.rectangle(frame_mask, (pad, pad), (pad + int(width * 0.18), pad + int(height * 0.12)), 255, -1)
                cv2.rectangle(frame_mask, (width - pad - int(width * 0.18), pad), (width - pad, pad + int(height * 0.12)), 255, -1)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            frame_mask = cv2.morphologyEx(frame_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            frame_mask = cv2.dilate(frame_mask, kernel, iterations=1)

            mask_path = output_dir / f"mask_{frame_index:05d}.png"
            cv2.imwrite(str(mask_path), frame_mask)
            written_masks += 1

        capture.release()

        if written_masks == 0:
            raise RuntimeError(f"No frames were processed for mask generation from {input_video_path}")

        logger.info("Dynamic mask sequence saved to %s (%d masks)", output_dir, written_masks)
        return str(output_dir)

    def remove_moving_watermark(
        self,
        input_video_path: str,
        output_video_path: str,
    ) -> str:
        input_path = Path(input_video_path).expanduser().resolve()
        output_path = Path(output_video_path).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_video_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video file: {input_video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        prev_gray: Optional[np.ndarray] = None
        frame_index = 0
        stability_mask: Optional[np.ndarray] = None

        while True:
            ret, frame = capture.read()
            if not ret:
                break

            frame_index += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (9, 9), 0)

            if prev_gray is None:
                prev_gray = blur
                writer.write(frame)
                continue

            mask = self._create_motion_mask(prev_gray, blur)
            logo_mask = self._extract_logo_region(mask)

            if stability_mask is None:
                stability_mask = logo_mask
            else:
                stability_mask = cv2.bitwise_or(stability_mask, logo_mask)
                stability_mask = cv2.morphologyEx(
                    stability_mask,
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)),
                    iterations=1,
                )

            if np.count_nonzero(logo_mask) == 0:
                inpaint_mask = stability_mask
            else:
                inpaint_mask = logo_mask

            text_mask = self._extract_text_mask(frame)
            yolo_mask = self._run_yolo(frame)
            combined_mask = cv2.bitwise_or(inpaint_mask, text_mask)
            combined_mask = cv2.bitwise_or(combined_mask, yolo_mask)
            if np.count_nonzero(combined_mask) > 0:
                inpaint_frame = cv2.inpaint(frame, combined_mask, self.inpaint_radius, self._get_inpaint_flag())
            else:
                inpaint_frame = frame

            writer.write(inpaint_frame)
            prev_gray = blur

        capture.release()
        writer.release()
        logger.info("Watermark/logo removal completed: %s", output_path)
        return str(output_path)
