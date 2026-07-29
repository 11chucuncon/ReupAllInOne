from __future__ import annotations

from typing import Any, Dict

from app.plugins.base import BaseStep


class TranslateStep(BaseStep):
    name = "AI Translate"

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context["translated_text"] = "Translated text"
        return context
