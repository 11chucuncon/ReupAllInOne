from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


class LLMRewriter:
    """Rewrite text using Google Gemini via the generative AI SDK."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config_path = config_path or str(Path(__file__).resolve().parents[1] / "config" / "settings.yaml")
        self.settings = self._load_settings()
        self.gemini_config = self.settings.get("gemini", {})
        self.api_key = self.gemini_config.get("api_key", "")
        self.model_name = self.gemini_config.get("model_name", "gemini-1.5-flash")
        self.prompt_template = self.gemini_config.get("prompt_template", "")
        self.model = None

        if self.api_key and self.api_key != "YOUR_GEMINI_API_KEY":
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
            except Exception as exc:
                logger.warning("Unable to initialize Gemini client: %s", exc)

    def _load_settings(self) -> dict:
        """Load configuration from config/settings.yaml."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
        except FileNotFoundError as exc:
            logger.error("Settings file not found at %s", self.config_path)
            raise FileNotFoundError(f"Missing configuration file: {self.config_path}") from exc
        except yaml.YAMLError as exc:
            logger.error("Failed to parse settings YAML: %s", exc)
            raise RuntimeError("Invalid YAML configuration") from exc

    def rewrite(self, text: str) -> str:
        """Rewrite input text via Gemini or fall back to the original text on failure."""
        if not text or not text.strip():
            return text

        if self.model is None:
            logger.warning("Gemini model is unavailable; returning original text")
            return text

        try:
            prompt = f"{self.prompt_template}\n\n{text}"
            response = self.model.generate_content(prompt)
            rewritten_text = getattr(response, "text", None) or str(response)
            if rewritten_text and rewritten_text.strip():
                return rewritten_text.strip()
            logger.warning("Gemini returned empty content; using original text")
            return text
        except Exception as exc:
            logger.warning("Gemini API call failed, using original text as fallback: %s", exc)
            return text
