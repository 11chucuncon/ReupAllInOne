from __future__ import annotations

from typing import Any, Dict

from app.plugins.base import BaseStep


class WatermarkStep(BaseStep):
    name = "Add Watermark"

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context["watermark_applied"] = True
        return context
