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
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the Gemini client if a valid API key is provided."""
        self.model = None
        api_key = (self.api_key or "").strip()
        if not api_key or api_key == "YOUR_GEMINI_API_KEY":
            return

        try:
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name)
        except Exception as exc:
            logger.warning("Unable to initialize Gemini client: %s", exc)

    def set_api_key(self, api_key: Optional[str]) -> None:
        """Set or update the Gemini API key and reinitialize the client."""
        self.api_key = (api_key or "").strip() or (self.gemini_config.get("api_key", "") or "").strip()
        self._initialize_client()

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
        """Rewrite input text via Gemini or raise a clear error if the API key is missing."""
        if not text or not text.strip():
            return text

        if self.model is None:
            raise RuntimeError(
                "Gemini API key is missing. Please enter a valid gemini API key before enabling AI rewrite."
            )

        try:
            prompt = f"{self.prompt_template}\n\n{text}"
            response = self.model.generate_content(prompt)
            rewritten_text = getattr(response, "text", None) or str(response)
            if rewritten_text and rewritten_text.strip():
                return rewritten_text.strip()
            logger.warning("Gemini returned empty content")
            raise RuntimeError("Gemini returned empty content")
        except Exception as exc:
            raise RuntimeError(f"Gemini API call failed: {exc}") from exc
