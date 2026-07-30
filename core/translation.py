from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class TranslationEngine:
    """Perform translation for subtitle text and prepare target-language pipelines."""

    def __init__(self, config_path: str | None = None) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.config_path = config_path or str(self.project_root / "config" / "settings.yaml")
        self.settings = self._load_settings()
        self.translation_config = self.settings.get("translation", {})
        self.provider = self.translation_config.get("provider", "gg")
        self.target_language = self.translation_config.get("target_language", "en")
        self.api_key = self.translation_config.get("api_key", "")

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

    def translate_text(self, text: str, target_language: str | None = None, api_key: str | None = None) -> str:
        target_language = target_language or self.target_language
        provider = (self.provider or "gg").lower()
        api_key = api_key or self.api_key or os.environ.get("GOOGLE_API_KEY")

        if provider == "local":
            return self._translate_local(text, target_language)
        if provider in {"gg", "google", "gemini"}:
            return self._translate_gg(text, target_language, api_key)
        return self._translate_openrouter(text, target_language, api_key)

    def _translate_local(self, text: str, target_language: str) -> str:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError("transformers is required for local translation") from exc

        translator = pipeline("translation", model=self.translation_config.get("model", "Helsinki-NLP/opus-mt-en-vi"))
        translations = translator(text, max_length=4096)
        return translations[0]["translation_text"]

    def _translate_gg(self, text: str, target_language: str, api_key: str | None = None) -> str:
        try:
            import google.generativeai as gg
            from google.generativeai import TextTranslation
        except ImportError as exc:
            raise RuntimeError("google-generativeai is required for GG translation provider") from exc

        if api_key:
            gg.configure(api_key=api_key)

        if not api_key:
            raise RuntimeError("Google Gemini API key is required for GG translation provider")

        return TextTranslation().translate(text=text, target_language=target_language)

    def _translate_openrouter(self, text: str, target_language: str, api_key: str | None = None) -> str:
        try:
            import json
            import urllib.request
        except ImportError as exc:
            raise RuntimeError("urllib is required for OpenRouter translation") from exc

        if not api_key:
            raise RuntimeError("OpenRouter API key is required for OpenRouter translation provider")

        api_url = "https://openrouter.ai/api/v1/chat/completions"
        prompt = (
            f"Hãy dịch đoạn văn sau sang {target_language} một cách tự nhiên và ngắn gọn:\n\n{text}"
        )
        payload = {
            "model": self.translation_config.get("model", "deepseek/deepseek-v4-flash"),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        request = urllib.request.Request(api_url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))

        text_output = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not text_output:
            raise RuntimeError("OpenRouter translation returned empty response")
        return text_output.strip()
