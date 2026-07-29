from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict
from urllib import request

from app.plugins.base import BaseStep


class TranslationStep(BaseStep):
    name = "Translation"

    def _load_env_file(self) -> None:
        env_path = Path(__file__).resolve().parents[2] / ".env"
        if not env_path.exists():
            return
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if key:
                os.environ[key] = value

    def _call_openrouter(self, api_key: str, model: str, text: str, target_language: str, config: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": f"Translate the following text to {target_language}. Return only the translated text.",
                },
                {"role": "user", "content": text},
            ],
            "temperature": config.get("temperature", 0.2),
            "max_tokens": config.get("max_tokens", 256),
            "stream": False,
        }
        api_base = self._resolve_config_value(config.get("api_base")) or "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "HTTP-Referer": self._resolve_config_value(config.get("http_referer")) or "https://github.com",
            "X-Title": self._resolve_config_value(config.get("app_name")) or "Video Pipeline",
            "User-Agent": "Mozilla/5.0",
        }
        req = request.Request(
            api_base,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=config.get("timeout", 30)) as response:
                data = json.loads(response.read().decode("utf-8"))
        except request.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"OpenRouter HTTP {exc.code}: {body}") from exc

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {"translated_text": content.strip(), "status": "translated"}

    def _resolve_config_value(self, value: Any) -> Any:
        if isinstance(value, str):
            if value.startswith("${") and value.endswith("}"):
                env_name = value[2:-1]
                return os.getenv(env_name, value)
            return value
        return value

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self._load_env_file()
        config = self.config or {}
        provider = self._resolve_config_value(config.get("provider", "llm"))
        target_language = config.get("target_language", "vi")
        text = context.get("asr_result", {}).get("text", "") or context.get("text", "")

        target_language = self._resolve_config_value(config.get("target_language", "vi"))

        translation_result = {
            "provider": provider,
            "target_language": target_language,
            "source_text": text,
            "translated_text": f"[{target_language}] {text or 'sample text'}",
            "status": "prepared",
        }

        if provider in {"openrouter", "openai", "qwen"}:
            api_key = self._resolve_config_value(config.get("api_key")) or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("QWEN_API_KEY")
            model = self._resolve_config_value(config.get("model")) or "google/gemini-2.5-flash-lite"
            print(f"[TranslationStep] provider={provider} model={model} api_key_present={bool(api_key)}")
            if api_key:
                try:
                    result = self._call_openrouter(api_key, model, text, target_language, config)
                    translation_result.update(result)
                    translation_result["engine_note"] = f"{provider} via OpenRouter configured"
                except Exception as exc:  # noqa: BLE001
                    translation_result["engine_note"] = f"Translation call failed: {exc}"
                    translation_result["status"] = "failed"
            else:
                translation_result["engine_note"] = f"{provider} API key not configured; using fallback"
        else:
            translation_result["engine_note"] = "Use a local LLM provider"

        context["translation_result"] = translation_result
        return context
