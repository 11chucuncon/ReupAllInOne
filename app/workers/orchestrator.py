from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from celery import group
from celery.exceptions import Retry

from app.workers.tasks import (
    audio_processing_task,
    download_task,
    render_task,
    transcribe_task,
    translate_task,
)


class CeleryOrchestrator:
    def run(self, source_url: str, output_dir: str | Path | None = None) -> Dict[str, Any]:
        output_path = Path(output_dir or "outputs")
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            download_result = download_task.delay(source_url, str(output_path))
            download_output = download_result.get(timeout=5)

            audio_result = audio_processing_task.delay(download_output["source_path"], str(output_path))
            audio_output = audio_result.get(timeout=5)

            transcript_result = transcribe_task.delay(audio_output["vocal_path"], str(output_path))
            transcript_output = transcript_result.get(timeout=5)

            translate_result = translate_task.delay(transcript_output["transcript"], str(output_path))
            translate_output = translate_result.get(timeout=5)

            render_result = render_task.delay(download_output["source_path"], str(output_path / "tts.wav"), str(output_path))

            return {
                "status": "queued",
                "download_task_id": download_result.id,
                "audio_task_id": audio_result.id,
                "transcribe_task_id": transcript_result.id,
                "translate_task_id": translate_result.id,
                "render_task_id": render_result.id,
                "output_dir": str(output_path),
                "download_output": download_output,
                "translate_output": translate_output,
            }
        except Exception as exc:
            return {"status": "fallback", "error": str(exc)}
