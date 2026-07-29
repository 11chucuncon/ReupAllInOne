from __future__ import annotations

from typing import Any, Dict, Optional


class BaseStep:
    name = "Base Step"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = config or {}

    def prepare_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        prepared = dict(context)
        prepared.update(self.config)
        return prepared

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError
