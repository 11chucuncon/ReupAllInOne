from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import yaml

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency
    YOLO = None

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


class YOLOv8SegmentationDetector:
    """High-quality YOLOv8 segmentation detector for subtitles, logos, and watermarks."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.settings = _load_settings(config_path)
        self.cleaner_config = self.settings.get("cleaner", {})
        self.confidence = float(self.cleaner_config.get("yolo_confidence", 0.35))
        self.iou_threshold = float(self.cleaner_config.get("yolo_nms_threshold", 0.45))
        self.model_name = self.cleaner_config.get("yolo_model", "yolov8n-seg.pt")
        self.model_path = Path(self.cleaner_config.get("yolo_model_path", "weights/yolov8n-seg.pt"))
        self._model: Optional[Any] = None

    def _resolve_model_path(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return Path(__file__).resolve().parents[1] / path

    def load_model(self) -> Optional[Any]:
        if YOLO is None:
            logger.warning("ultralytics is not installed; mask generation will fall back to contour-based detection")
            return None
        if self._model is not None:
            return self._model

        model_path = self._resolve_model_path(self.model_path)
        try:
            if model_path.exists():
                self._model = YOLO(str(model_path))
            else:
                self._model = YOLO(self.model_name)
            logger.info("Loaded YOLOv8 segmentation model %s", self.model_name)
            return self._model
        except Exception as exc:  # pragma: no cover - runtime dependency
            logger.warning("YOLOv8 segmentation model initialization failed: %s", exc)
            return None

    def detect_mask(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        model = self.load_model()
        if model is None:
            return mask

        try:
            results = model(frame, stream=False, imgsz=1280, conf=self.confidence, iou=self.iou_threshold, verbose=False)
        except Exception as exc:  # pragma: no cover - runtime dependency
            logger.warning("YOLOv8 segmentation inference failed: %s", exc)
            return mask

        for result in results:
            masks = getattr(result, "masks", None)
            if masks is None:
                continue
            mask_tensor = getattr(masks, "data", None)
            if mask_tensor is None:
                continue
            for item in mask_tensor:
                try:
                    candidate = item.cpu().numpy() if hasattr(item, "cpu") else np.asarray(item)
                except Exception:
                    continue
                if candidate.ndim == 3:
                    candidate = candidate[0]
                candidate = np.asarray(candidate)
                if candidate.size == 0:
                    continue
                if candidate.shape != (height, width):
                    try:
                        resized = cv2.resize(candidate.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
                    except Exception:
                        resized = candidate.astype(np.uint8)
                else:
                    resized = candidate.astype(np.uint8)
                mask |= (resized > 0.5).astype(np.uint8) * 255

        if np.count_nonzero(mask) > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=1)
        return mask
