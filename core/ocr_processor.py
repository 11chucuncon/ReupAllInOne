from __future__ import annotations

import logging
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)


def _flatten_to_string_list(val, default: str = "en") -> list[str]:
    """Flatten nested containers (set/list/tuple) or single values into a list[str].

    Maps common Chinese codes to EasyOCR expected codes.
    """
    result: list[str] = []
    stack = [val]
    while stack:
        curr = stack.pop()
        if isinstance(curr, (list, tuple, set)):
            # extend using list so sets get expanded
            stack.extend(list(curr))
        elif curr is None:
            continue
        else:
            s = str(curr).strip()
            if s:
                result.append(s)

    if not result:
        result = [default]

    easyocr_map = {
        "zh": "zh_sim",
        "zh-cn": "zh_sim",
        "zh_cn": "zh_sim",
        "zh-tw": "zh_tra",
        "zh_tw": "zh_tra",
    }
    normalized: list[str] = []
    for item in result:
        key = item.lower()
        mapped = easyocr_map.get(key, item)
        if mapped not in normalized:
            normalized.append(mapped)
    return normalized


class OCRProcessor:
    """Detect text, bounding boxes, and generate OCR segments from video frames."""

    def __init__(self, config_path: str | None = None, languages: Any | None = None) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.config_path = config_path or str(self.project_root / "config" / "settings.yaml")
        self.settings = self._load_settings()
        self.ocr_config = self.settings.get("ocr", {})
        # Determine raw languages: prefer explicitly passed `languages`, fall back to config
        if languages is not None:
            raw_langs = languages
        else:
            raw_langs = self.ocr_config.get("languages", ["vi", "en", "zh", "ja", "ko"])

        # Flatten and normalize to a list[str] suitable for easyocr.Reader
        self.languages = _flatten_to_string_list(raw_langs, default="en")
        self.sample_rate = int(self.ocr_config.get("sample_rate", 1))

    def _load_settings(self) -> dict[str, Any]:
        try:
            with open(self.config_path, "r", encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
        except FileNotFoundError as exc:
            logger.error("Settings file not found at %s", self.config_path)
            raise
        except yaml.YAMLError as exc:
            logger.error("Invalid YAML configuration: %s", exc)
            raise

    def _run_command(self, command: list[str]) -> None:
        logger.info("Running command: %s", " ".join(shlex.quote(part) for part in command))
        subprocess.run(command, check=True)

    def _get_video_resolution(self, video_path: str) -> tuple[int, int]:
        command = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            video_path,
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        width, height = result.stdout.strip().split("x")
        return int(width), int(height)

    def extract_frames(self, video_path: str, output_dir: Path) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_pattern = str(output_dir / "frame_%04d.png")
        command = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vf",
            f"fps={self.sample_rate},scale=960:-1",
            frame_pattern,
        ]
        self._run_command(command)
        return sorted(output_dir.glob("frame_*.png"))

    def detect_text(self, video_path: str) -> dict[str, Any]:
        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError("easyocr and opencv-python-headless are required for OCR detection") from exc

        reader = easyocr.Reader(self.languages, gpu=False)
        original_width, original_height = self._get_video_resolution(video_path)

        with tempfile.TemporaryDirectory(prefix="reup_ocr_") as temp_dir:
            frame_dir = Path(temp_dir) / "frames"
            frames = self.extract_frames(video_path, frame_dir)
            segments: list[dict[str, Any]] = []

            for index, frame_path in enumerate(frames, start=1):
                image_results = reader.readtext(str(frame_path), detail=1)
                frame_image = cv2.imread(str(frame_path))
                if frame_image is None:
                    continue
                scaled_height, scaled_width = frame_image.shape[:2]
                ratio_x = original_width / float(scaled_width)
                ratio_y = original_height / float(scaled_height)

                for text_data in image_results:
                    bbox, text, _ = text_data
                    if not text.strip():
                        continue
                    original_bbox = [
                        [int(round(x * ratio_x)), int(round(y * ratio_y))]
                        for x, y in bbox
                    ]
                    segments.append(
                        {
                            "frame": frame_path.name,
                            "index": len(segments) + 1,
                            "text": text,
                            "bbox": original_bbox,
                            "start_time": (index - 1) / float(self.sample_rate),
                            "end_time": index / float(self.sample_rate),
                        }
                    )

        transcript = "\n".join(item["text"] for item in segments)
        return {
            "full_text": transcript,
            "segments": segments,
        }

    def _expand_bbox(self, bbox: list[list[int]], original_width: int, original_height: int, padding_ratio: float = 0.05) -> tuple[int, int, int, int]:
        x_coords = [pt[0] for pt in bbox]
        y_coords = [pt[1] for pt in bbox]
        x0 = max(0, int(min(x_coords) - original_width * padding_ratio))
        x1 = min(original_width, int(max(x_coords) + original_width * padding_ratio))
        y0 = max(0, int(min(y_coords) - original_height * padding_ratio))
        y1 = min(original_height, int(max(y_coords) + original_height * padding_ratio))
        return x0, y0, x1, y1

    def build_mask_image(
        self,
        video_path: str,
        output_path: str,
        segments: list[dict[str, Any]],
        auto_remove_watermark: bool = False,
    ) -> Optional[str]:
        if not segments and not auto_remove_watermark:
            return None

        original_width, original_height = self._get_video_resolution(video_path)
        mask = np.zeros((original_height, original_width), dtype=np.uint8)

        for segment in segments:
            bbox = segment.get("bbox")
            if not bbox or not isinstance(bbox, list):
                continue
            pts = np.array(bbox, dtype=np.int32)
            if pts.shape == (4, 2):
                x0, y0, x1, y1 = self._expand_bbox(bbox, original_width, original_height)
                cv2.rectangle(mask, (x0, y0), (x1, y1), color=255, thickness=-1)
            else:
                x_coords = pts[:, 0]
                y_coords = pts[:, 1]
                x0, x1 = int(np.min(x_coords)), int(np.max(x_coords))
                y0, y1 = int(np.min(y_coords)), int(np.max(y_coords))
                x0, y0, x1, y1 = self._expand_bbox([[x0, y0], [x1, y1]], original_width, original_height)
                cv2.rectangle(mask, (x0, y0), (x1, y1), color=255, thickness=-1)

        if auto_remove_watermark:
            corner_width = int(original_width * 0.18)
            corner_height = int(original_height * 0.12)
            margin = int(original_width * 0.02)
            top_left = (margin, margin, corner_width, corner_height)
            top_right = (original_width - corner_width - margin, margin, original_width - margin, corner_height)
            bottom_left = (margin, original_height - corner_height - margin, corner_width, original_height - margin)
            bottom_right = (
                original_width - corner_width - margin,
                original_height - corner_height - margin,
                original_width - margin,
                original_height - margin,
            )
            for rect in (top_left, top_right, bottom_left, bottom_right):
                x0, y0, x1, y1 = rect
                cv2.rectangle(mask, (x0, y0), (x1, y1), color=255, thickness=-1)

            strip_height = int(original_height * 0.12)
            cv2.rectangle(mask, (0, original_height - strip_height), (original_width, original_height), color=255, thickness=-1)

        output_image = Path(output_path)
        output_image.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_image), mask)
        return str(output_image)

    def generate_subtitle_overlays(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return segments
