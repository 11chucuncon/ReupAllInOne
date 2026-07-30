from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)


class LLMRewriter:
    """Rewrite text using the OpenRouter API."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config_path = config_path or str(Path(__file__).resolve().parents[1] / "config" / "settings.yaml")
        self.settings = self._load_settings()
        self.openrouter_config = self.settings.get("openrouter", self.settings.get("gemini", {}))
        self.api_key = self.openrouter_config.get("api_key", "")
        self.model_name = self.openrouter_config.get("model_name", "deepseek/deepseek-v4-flash")
        self.prompt_template = self.openrouter_config.get("prompt_template", "")
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Validate that an API key is available for the OpenRouter client."""
        api_key = (self.api_key or "").strip()
        self.ready = bool(api_key) and api_key != "YOUR_OPENROUTER_API_KEY"

    def set_api_key(self, api_key: Optional[str]) -> None:
        """Set or update the OpenRouter API key."""
        self.api_key = (api_key or "").strip() or (self.openrouter_config.get("api_key", "") or "").strip()
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

    def rewrite(self, text: str, target_lang: str = "vi") -> str:
        """Rewrite and translate input text via the OpenRouter API or raise a clear error if the API key is missing."""
        if not text or not text.strip():
            return text

        if not self.ready:
            raise RuntimeError(
                "OpenRouter API key is missing. Please enter a valid OpenRouter API key before enabling AI rewrite."
            )

        try:
            lang_code = (target_lang or "vi").split()[0].split("(")[0].strip()
            prompt = (
                f"Hãy dịch và viết lại kịch bản sau đây sang {lang_code} theo phong cách cuốn hút, tự nhiên, "
                f"chuẩn văn phong video ngắn ngắn (Shorts/TikTok). Chỉ trả về nội dung kịch bản bằng {lang_code}, không kèm giải thích:\n\n{text}"
            )
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/11chucuncon/ReupAllInOne",
                "X-Title": "AI Video Reup Studio",
            }

            request = urllib.request.Request(self.api_url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))

            rewritten_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if rewritten_text and rewritten_text.strip():
                return rewritten_text.strip()
            logger.warning("OpenRouter returned empty content")
            raise RuntimeError("OpenRouter returned empty content")
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"OpenRouter API request failed: {exc.read().decode('utf-8', errors='ignore')}") from exc
        except Exception as exc:
            raise RuntimeError(f"OpenRouter API call failed: {exc}") from exc
