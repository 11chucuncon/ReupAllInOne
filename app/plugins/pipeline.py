from __future__ import annotations

from typing import Any, Dict, List

from app.plugins.base import BaseStep


class VideoPipeline:
    def __init__(self, active_steps: List[BaseStep]) -> None:
        self.steps = active_steps

    def run(self, initial_data: Dict[str, Any]) -> Dict[str, Any]:
        context = initial_data
        # optional progress callback passed through context
        progress = context.get("progress_callback")
        for step in self.steps:
            prepared_context = step.prepare_context(context)
            msg = f"start:{step.name}"
            if callable(progress):
                try:
                    progress(msg)
                except Exception:
                    pass
            else:
                print(f"[Pipeline] start -> {step.name}")

            context = step.execute(prepared_context)

            if step.name == "Download Video":
                status_msg = f"download:{context.get('download_status')}"
                if callable(progress):
                    try:
                        progress(status_msg)
                    except Exception:
                        pass
                else:
                    print(f"[Pipeline] download ok -> {context.get('download_status')}")
            elif step.name == "ASR":
                status_msg = f"asr:{context.get('asr_result', {}).get('text')}"
                if callable(progress):
                    try:
                        progress(status_msg)
                    except Exception:
                        pass
                else:
                    print(f"[Pipeline] asr ok -> {context.get('asr_result', {}).get('text')}")
            elif step.name == "Translation":
                status_msg = f"translate:{context.get('translation_result', {}).get('translated_text')}"
                if callable(progress):
                    try:
                        progress(status_msg)
                    except Exception:
                        pass
                else:
                    print(f"[Pipeline] translate ok -> {context.get('translation_result', {}).get('translated_text')}")
            elif step.name == "FFmpeg Render":
                status_msg = f"render:{context.get('render_status')}"
                if callable(progress):
                    try:
                        progress(status_msg)
                    except Exception:
                        pass
                else:
                    print(f"[Pipeline] render ok -> {context.get('render_status')}")

            end_msg = f"end:{step.name}"
            if callable(progress):
                try:
                    progress(end_msg)
                except Exception:
                    pass
            else:
                print(f"[Pipeline] end -> {step.name}")
        return context
