from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from app.modules.audio_processor import AudioProcessor
from app.modules.asr import ASRProcessor
from app.modules.downloader import VideoDownloader
from app.modules.renderer import Renderer
from app.modules.translator import Translator
from app.modules.tts import TTSProcessor
from app.workers.orchestrator import CeleryOrchestrator


def build_pipeline_plan(source_url: str) -> Dict[str, Any]:
    """Build the initial processing plan for a video source."""
    steps: List[Dict[str, str]] = [
        {"name": "download", "description": "Download source video"},
        {"name": "audio_separation", "description": "Separate vocals and instrumental"},
        {"name": "transcribe", "description": "Transcribe speech to text"},
        {"name": "translate", "description": "Translate and normalize subtitles"},
        {"name": "render", "description": "Render final video"},
    ]
    return {"source_url": source_url, "steps": steps}


def run_pipeline(source_url: str, output_dir: str | Path | None = None) -> Dict[str, Any]:
    """Run a simple end-to-end pipeline in a single process or via Celery worker."""
    output_path = Path(output_dir or "outputs")
    output_path.mkdir(parents=True, exist_ok=True)

    orchestrator = CeleryOrchestrator()
    celery_result = orchestrator.run(source_url, output_path)
    if celery_result["status"] in {"queued", "fallback"}:
        if celery_result["status"] == "queued":
            return celery_result

    downloader = VideoDownloader(output_dir=output_path)
    audio_processor = AudioProcessor()
    asr = ASRProcessor()
    translator = Translator()
    tts = TTSProcessor()
    renderer = Renderer()

    source_path = downloader.download(source_url, output_path)
    audio_outputs = audio_processor.process(source_path, output_path)
    transcript = asr.transcribe(audio_outputs["vocal"], output_path)
    translated = translator.translate(transcript)
    tts_audio = tts.synthesize(translated, output_path)
    output_path_final = renderer.render(source_path, tts_audio, output_path)

    return {
        "status": "completed",
        "source_url": source_url,
        "output_path": output_path_final,
        "transcript": transcript,
        "translated": translated,
    }
