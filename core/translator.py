from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional, Sequence

import requests
import yaml

logger = logging.getLogger(__name__)


class TranslationEngine:
    """Translate text using OpenRouter LLM with a safe Google Translate fallback."""

    def __init__(self, config_path: str | None = None) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.config_path = config_path or str(self.project_root / "config" / "settings.yaml")
        self.settings = self._load_settings()
        self.translation_config = self.settings.get("translation", {})
        self.provider = str(self.translation_config.get("provider", "openrouter")).lower()
        self.target_language = self.translation_config.get("target_language", "vi")
        self.api_key = str(self.translation_config.get("api_key", "") or "")
        self.openrouter_model = self._resolve_openrouter_model()

    def _load_settings(self) -> dict[str, Any]:
        try:
            with open(self.config_path, "r", encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
        except FileNotFoundError:
            logger.warning("Settings file not found at %s; using empty defaults", self.config_path)
            return {}
        except yaml.YAMLError as exc:
            logger.warning("Invalid YAML configuration: %s", exc)
            return {}

    def _resolve_openrouter_model(self) -> str:
        """Pick a valid OpenRouter model and avoid legacy Hugging Face translation model names."""
        configured_model = str(
            self.translation_config.get("openrouter_model")
            or self.translation_config.get("model")
            or os.environ.get("OPENROUTER_MODEL", "")
            or ""
        ).strip()

        if self.provider in {"openrouter", "llm", "deepseek"}:
            if not configured_model:
                return "deepseek/deepseek-chat"
            normalized = configured_model.lower()
            if "opus-mt" in normalized or "huggingface" in normalized or normalized.startswith("helsinki"):
                logger.warning(
                    "Detected a Hugging Face-style translation model '%s'; using OpenRouter model 'deepseek/deepseek-chat' instead.",
                    configured_model,
                )
                return "deepseek/deepseek-chat"
            return configured_model

        return configured_model or "deepseek/deepseek-chat"

    def translate_text(self, text: str, target_language: str | None = None, api_key: str | None = None) -> str:
        """Translate a single text chunk with OpenRouter and graceful fallback."""
        if not text or not str(text).strip():
            return ""

        target_language = target_language or self.target_language
        resolved_api_key = api_key or self.api_key or os.environ.get("OPENROUTER_API_KEY") or ""

        if self.provider in {"openrouter", "llm", "deepseek"}:
            try:
                return self._translate_with_openrouter(str(text).strip(), target_language, resolved_api_key)
            except Exception as exc:
                logger.warning("OpenRouter translation failed, falling back to Google Translate: %s", exc)
                return self._translate_with_google_fallback(str(text).strip(), target_language)

        if self.provider in {"gg", "google", "gemini"}:
            try:
                return self._translate_with_google_fallback(str(text).strip(), target_language)
            except Exception as exc:
                logger.warning("Google Translate fallback failed: %s", exc)
                return str(text).strip()

        return self._translate_with_google_fallback(str(text).strip(), target_language)

    def translate_batch(self, texts: Sequence[str], target_language: str | None = None, api_key: str | None = None) -> list[str]:
        """Translate a list of texts in a single request using a JSON-array batch prompt."""
        cleaned_texts = [str(text).strip() for text in texts if str(text).strip()]
        if not cleaned_texts:
            return []

        if len(cleaned_texts) == 1:
            return [self.translate_text(cleaned_texts[0], target_language=target_language, api_key=api_key)]

        if not self._should_use_batch(cleaned_texts):
            return [self.translate_text(text, target_language=target_language, api_key=api_key) for text in cleaned_texts]

        target_language = target_language or self.target_language
        resolved_api_key = api_key or self.api_key or os.environ.get("OPENROUTER_API_KEY") or ""

        try:
            return self._translate_batch_with_openrouter(cleaned_texts, target_language, resolved_api_key)
        except Exception as exc:
            logger.warning("Batch OpenRouter translation failed, falling back per-item: %s", exc)
            return [self.translate_text(text, target_language=target_language, api_key=resolved_api_key) for text in cleaned_texts]

    def translate_segments(
        self,
        segments: Sequence[dict[str, Any]],
        target_language: str | None = None,
        api_key: str | None = None,
    ) -> list[dict[str, Any]]:
        """Translate subtitle-like segments while preserving the original start/end timecodes."""
        texts = [str(segment.get("text", "")).strip() for segment in segments if str(segment.get("text", "")).strip()]
        translated_texts = self.translate_batch(texts, target_language=target_language, api_key=api_key)

        translated_segments: list[dict[str, Any]] = []
        for index, segment in enumerate(segments):
            text = str(segment.get("text", "")).strip()
            if not text:
                continue
            translated_segments.append({
                "start": float(segment.get("start", 0.0)),
                "end": float(segment.get("end", segment.get("start", 0.0))),
                "text": translated_texts[min(index, len(translated_texts) - 1)] if translated_texts else text,
            })
        return translated_segments

    def _should_use_batch(self, texts: Sequence[str]) -> bool:
        """Return True only for a compact batch payload that is likely to succeed."""
        if len(texts) > 8:
            return False
        if any(len(text) > 700 for text in texts):
            return False
        total_chars = sum(len(text) for text in texts)
        return total_chars <= 3000

    def _translate_with_openrouter(self, text: str, target_language: str, api_key: str) -> str:
        if not api_key:
            raise RuntimeError("OpenRouter API key is required")

        payload = {
            "model": self.openrouter_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a professional video translator. Translate the given text into natural, fluent "
                        f"{self._display_language_name(target_language)} suitable for voiceover/dubbing. "
                        "Keep the translation concise so it fits the timing of the original speech. "
                        "Output ONLY the translated text without explanation or quotes."
                    ),
                },
                {"role": "user", "content": text},
            ],
            "temperature": 0.2,
            "max_tokens": 1024,
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/11chucuncon/ReupAllInOne",
                "X-Title": "AI Video Reup Studio",
            },
            json=payload,
            timeout=60,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"OpenRouter request failed ({response.status_code}): {response.text[:500]}")

        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            raise RuntimeError("OpenRouter translation returned an empty response")
        return str(content).strip()

    def _translate_batch_with_openrouter(self, texts: Sequence[str], target_language: str, api_key: str) -> list[str]:
        if not api_key:
            raise RuntimeError("OpenRouter API key is required")

        payload_text = [{"id": index + 1, "text": text} for index, text in enumerate(texts)]
        payload = {
            "model": self.openrouter_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a video subtitle translator. Translate the array of texts into natural "
                        f"{self._display_language_name(target_language)}. Maintain context and tone across all lines. "
                        "Return ONLY a JSON list of translated strings with the exact same order and length."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Translate this JSON array of subtitles:\n{payload_text}",
                },
            ],
            "temperature": 0.2,
            "max_tokens": 2048,
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/11chucuncon/ReupAllInOne",
                "X-Title": "AI Video Reup Studio",
            },
            json=payload,
            timeout=60,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"OpenRouter batch request failed ({response.status_code}): {response.text[:500]}")

        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            raise RuntimeError("OpenRouter batch translation returned an empty response")

        parsed = self._parse_json_translation_response(content)
        if parsed is None:
            raise RuntimeError("OpenRouter batch response was not valid JSON")
        if len(parsed) != len(texts):
            raise RuntimeError("OpenRouter batch response length did not match input size")
        return [str(item).strip() for item in parsed]

    def _parse_json_translation_response(self, content: str) -> Optional[list[str]]:
        """Parse a JSON array returned by the LLM and normalize it into a list of strings."""
        try:
            text = str(content).strip()
            if text.startswith("```"):
                text = text.strip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except Exception as exc:
            logger.warning("Could not parse OpenRouter JSON batch response: %s", exc)
        return None

    def _translate_with_google_fallback(self, text: str, target_language: str) -> str:
        try:
            from deep_translator import GoogleTranslator
        except ImportError:
            logger.warning("deep-translator is not installed; returning original text")
            return text

        translator = GoogleTranslator(source="auto", target=self._normalize_target_code(target_language))
        return translator.translate(text)

    def _normalize_target_code(self, target_language: str) -> str:
        normalized = str(target_language or "vi").strip().lower()
        mapping = {
            "vi": "vi",
            "vi-vn": "vi",
            "en": "en",
            "en-us": "en",
            "zh": "zh-cn",
            "zh-cn": "zh-cn",
            "zh-tw": "zh-tw",
            "ja": "ja",
            "ja-jp": "ja",
            "ko": "ko",
            "ko-kr": "ko",
        }
        return mapping.get(normalized, normalized)

    def _display_language_name(self, target_language: str) -> str:
        normalized = str(target_language or "vi").strip().lower()
        names = {
            "vi": "Vietnamese",
            "vi-vn": "Vietnamese",
            "en": "English",
            "en-us": "English",
            "zh": "Chinese",
            "zh-cn": "Chinese",
            "zh-tw": "Traditional Chinese",
            "ja": "Japanese",
            "ja-jp": "Japanese",
            "ko": "Korean",
            "ko-kr": "Korean",
        }
        return names.get(normalized, "the target language")
