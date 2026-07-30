from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import yaml

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
        self.text_score_threshold = float(self.cleaner_config.get("text_score_threshold", 0.5))
        self.text_nms_threshold = float(self.cleaner_config.get("text_nms_threshold", 0.4))
        self.yolo_config_path = Path(self.cleaner_config.get("yolo_config", "weights/yolov8n-watermark.cfg"))
        self.yolo_weights_path = Path(self.cleaner_config.get("yolo_weights", "weights/yolov8n-watermark.weights"))
        self.text_model_path = Path(self.cleaner_config.get("east_model", "weights/frozen_east_text_detection.pb"))

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

    def _extract_text_mask(self, frame: np.ndarray, text_net: Optional[cv2.dnn_Net] = None) -> np.ndarray:
        text_net = text_net if text_net is not None else self._load_text_model()
        text_mask = self._run_text_detector(frame, text_net)
        if np.count_nonzero(text_mask) > 0:
            return text_mask

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

    def _load_yolo_model(self) -> Optional[cv2.dnn_Net]:
        config_path = self._resolve_model_path(self.yolo_config_path)
        weights_path = self._resolve_model_path(self.yolo_weights_path)
        if not config_path.exists() or not weights_path.exists():
            logger.warning(
                "YOLO watermark model weight files not found at %s and %s; falling back to motion detection only",
                config_path,
                weights_path,
            )
            return None
        net = cv2.dnn.readNetFromDarknet(str(config_path), str(weights_path))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net

    def _run_yolo(self, frame: np.ndarray, net: Optional[cv2.dnn_Net]) -> np.ndarray:
        masks = np.zeros(frame.shape[:2], dtype=np.uint8)
        if net is None:
            return masks

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (640, 640), swapRB=True, crop=False)
        net.setInput(blob)
        layer_names = net.getUnconnectedOutLayersNames()
        outputs = net.forward(layer_names)

        height, width = frame.shape[:2]
        boxes: list[tuple[int, int, int, int]] = []
        confidences: list[float] = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                if scores.size == 0:
                    continue
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if confidence < self.yolo_confidence:
                    continue
                cx, cy, w, h = (detection[0:4] * np.array([width, height, width, height])).astype(int)
                x = int(cx - w / 2)
                y = int(cy - h / 2)
                boxes.append((x, y, int(w), int(h)))
                confidences.append(confidence)

        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.yolo_confidence, self.yolo_nms_threshold)
            for i in indices.flatten() if len(indices) else []:
                x, y, w, h = boxes[i]
                x0 = max(0, x)
                y0 = max(0, y)
                x1 = min(width, x + w)
                y1 = min(height, y + h)
                cv2.rectangle(masks, (x0, y0), (x1, y1), 255, -1)
        return masks

    def _load_text_model(self) -> Optional[cv2.dnn_Net]:
        model_path = self._resolve_model_path(self.text_model_path)
        if not model_path.exists():
            logger.warning("EAST text detection model not found at %s; using OpenCV fallback text segmentation", model_path)
            return None
        net = cv2.dnn.readNet(str(model_path))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net

    def _decode_text_predictions(self, scores: np.ndarray, geometry: np.ndarray, score_thresh: float) -> tuple[list[tuple[int, int, int, int]], list[float]]:
        rects: list[tuple[int, int, int, int]] = []
        confidences: list[float] = []

        height, width = scores.shape[2:4]
        for y in range(height):
            for x in range(width):
                score = float(scores[0, 0, y, x])
                if score < score_thresh:
                    continue
                offset_x = x * 4.0
                offset_y = y * 4.0
                angle = float(geometry[0, 4, y, x])
                cos = math.cos(angle)
                sin = math.sin(angle)
                h = float(geometry[0, 0, y, x] + geometry[0, 2, y, x])
                w = float(geometry[0, 1, y, x] + geometry[0, 3, y, x])
                end_x = int(offset_x + cos * geometry[0, 1, y, x] + sin * geometry[0, 2, y, x])
                end_y = int(offset_y - sin * geometry[0, 1, y, x] + cos * geometry[0, 2, y, x])
                start_x = int(end_x - w)
                start_y = int(end_y - h)
                rects.append((start_x, start_y, int(w), int(h)))
                confidences.append(score)
        return rects, confidences

    def _run_text_detector(self, frame: np.ndarray, net: Optional[cv2.dnn_Net]) -> np.ndarray:
        if net is None:
            return np.zeros(frame.shape[:2], dtype=np.uint8)

        orig_h, orig_w = frame.shape[:2]
        new_w, new_h = (320, 320)
        blob = cv2.dnn.blobFromImage(frame, 1.0, (new_w, new_h), (123.68, 116.78, 103.94), True, False)
        net.setInput(blob)
        scores, geometry = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

        rects, confidences = self._decode_text_predictions(scores, geometry, self.text_score_threshold)
        indices = cv2.dnn.NMSBoxes(rects, confidences, self.text_score_threshold, self.text_nms_threshold)
        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        if len(indices):
            for i in indices.flatten():
                x, y, w, h = rects[i]
                x0 = max(0, int(x * orig_w / new_w))
                y0 = max(0, int(y * orig_h / new_h))
                x1 = min(orig_w, int((x + w) * orig_w / new_w))
                y1 = min(orig_h, int((y + h) * orig_h / new_h))
                cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
        return mask

    def generate_combined_mask(self, input_video_path: str, output_mask_path: str) -> str:
        input_path = Path(input_video_path).expanduser().resolve()
        output_path = Path(output_mask_path).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_video_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video file: {input_video_path}")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        combined_mask = np.zeros((height, width), dtype=np.uint8)

        yolo_net = self._load_yolo_model()
        text_net = self._load_text_model()
        prev_gray: Optional[np.ndarray] = None
        frame_index = 0

        while True:
            ret, frame = capture.read()
            if not ret:
                break

            frame_index += 1
            if self.sample_rate > 1 and frame_index % self.sample_rate != 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (9, 9), 0)

            text_mask = self._extract_text_mask(frame, text_net)
            yolo_mask = self._run_yolo(frame, yolo_net)
            frame_mask = cv2.bitwise_or(text_mask, yolo_mask)

            if prev_gray is not None:
                motion_mask = self._create_motion_mask(prev_gray, blur)
                logo_mask = self._extract_logo_region(motion_mask)
                frame_mask = cv2.bitwise_or(frame_mask, logo_mask)

            prev_gray = blur
            combined_mask = cv2.bitwise_or(combined_mask, frame_mask)

        capture.release()

        if np.count_nonzero(combined_mask) == 0:
            logger.warning("No text or logo regions were detected. Falling back to conservative subtitle/watermark hotspots.")
            pad = int(min(width, height) * 0.02)
            bottom_strip = int(height * 0.12)
            cv2.rectangle(combined_mask, (pad, height - bottom_strip), (width - pad, height - pad), 255, -1)
            cv2.rectangle(combined_mask, (pad, pad), (pad + int(width * 0.18), pad + int(height * 0.12)), 255, -1)
            cv2.rectangle(combined_mask, (width - pad - int(width * 0.18), pad), (width - pad, pad + int(height * 0.12)), 255, -1)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)

        cv2.imwrite(str(output_path), combined_mask)
        logger.info("Combined detection mask saved to %s", output_path)
        return str(output_path)

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

        yolo_net = self._load_yolo_model()
        text_net = self._load_text_model()
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

            text_mask = self._extract_text_mask(frame, text_net)
            yolo_mask = self._run_yolo(frame, yolo_net)
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
