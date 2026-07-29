from __future__ import annotations

from typing import Any, Dict, List

from app.plugins.base import BaseStep


class VideoPipeline:
    def __init__(self, active_steps: List[BaseStep]) -> None:
        self.steps = active_steps

    def run(self, initial_data: Dict[str, Any]) -> Dict[str, Any]:
        context = initial_data
        for step in self.steps:
            prepared_context = step.prepare_context(context)
            print(f"[Pipeline] start -> {step.name}")
            context = step.execute(prepared_context)
            if step.name == "Download Video":
                print(f"[Pipeline] download ok -> {context.get('download_status')}")
            elif step.name == "ASR":
                print(f"[Pipeline] asr ok -> {context.get('asr_result', {}).get('text')}")
            elif step.name == "Translation":
                print(f"[Pipeline] translate ok -> {context.get('translation_result', {}).get('translated_text')}")
            elif step.name == "FFmpeg Render":
                print(f"[Pipeline] render ok -> {context.get('render_status')}")
            print(f"[Pipeline] end -> {step.name}")
        return context
