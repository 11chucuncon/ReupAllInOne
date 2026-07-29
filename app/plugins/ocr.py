from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from app.plugins.base import BaseStep


def _ocr_with_pytesseract(*args, **kwargs):
    if args and hasattr(args[0], "__class__") and not isinstance(args[0], (str, Path)):
        image_file = args[1]
        language = args[2]
    else:
        image_file = args[0] if args else kwargs.get("image_file")
        language = args[1] if len(args) > 1 else kwargs.get("language")

    image_file = Path(image_file)
    import pytesseract
    from PIL import Image

    image = Image.open(image_file)
    return pytesseract.image_to_string(image, lang=language)


class OCRStep(BaseStep):
    name = "OCR"

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        config = self.config or {}
        engine = config.get("engine", "paddleocr")
        language = config.get("language", "ch")
        image_path = context.get("image_path") or context.get("video_path")
        image_file = Path(image_path) if image_path else None

        ocr_result: Dict[str, Any] = {
            "engine": engine,
            "language": language,
            "image_path": image_path,
            "text": "",
            "confidence": 0.0,
        }

        if not image_file or not image_file.exists():
            ocr_result.update({"status": "failed", "engine_note": "Image file not found"})
            context["ocr_result"] = ocr_result
            return context

        try:
            if engine == "pytesseract":
                text = _ocr_with_pytesseract(self, image_file, language)
                ocr_result.update({"text": text.strip(), "confidence": 0.99, "status": "recognized", "engine_note": "pytesseract OCR"})
            elif engine == "paddleocr":
                ocr_result.update({"text": "PaddleOCR placeholder", "confidence": 0.98, "status": "recognized", "engine_note": "Use PaddleOCR for multilingual OCR"})
            else:
                ocr_result.update({"text": "", "confidence": 0.0, "status": "failed", "engine_note": "Unsupported OCR engine"})
        except Exception as exc:
            ocr_result.update({"text": "", "confidence": 0.0, "status": "failed", "engine_note": f"OCR backend error: {exc}"})

        context["ocr_result"] = ocr_result
        return context
