from __future__ import annotations

from pathlib import Path
import os

from celery import Celery
from celery.exceptions import Retry

broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

app = Celery("video_pipeline", broker=broker_url, backend=result_backend)
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_routes={"app.workers.tasks.*": {"queue": "video_pipeline"}},
    task_default_queue="video_pipeline",
    task_acks_late=True,
    broker_connection_retry_on_startup=True,
)


@app.task(bind=True, max_retries=3, default_retry_delay=2)
def download_task(self, source_url: str, output_dir: str) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    target = output_path / "source.mp4"
    target.write_bytes(b"mock video payload")
    return {"source_path": str(target)}


@app.task(bind=True, max_retries=3, default_retry_delay=2)
def audio_processing_task(self, source_path: str, output_dir: str) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    vocal_path = output_path / "vocal.wav"
    instrumental_path = output_path / "instrumental.wav"
    vocal_path.write_bytes(b"mock vocal")
    instrumental_path.write_bytes(b"mock instrumental")
    return {"vocal_path": str(vocal_path), "instrumental_path": str(instrumental_path)}


@app.task(bind=True, max_retries=3, default_retry_delay=2)
def transcribe_task(self, source_path: str, output_dir: str) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    transcript_path = output_path / "transcript.json"
    transcript_path.write_text('[{"start": "00:00:00", "end": "00:00:02", "text": "Hello world"}]', encoding="utf-8")
    return {"transcript": [{"start": "00:00:00", "end": "00:00:02", "text": "Hello world"}]}


@app.task(bind=True, max_retries=3, default_retry_delay=2)
def translate_task(self, transcript: list[dict], output_dir: str) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    translated = [{"start": item["start"], "end": item["end"], "text": f"Translated: {item['text']}"} for item in transcript]
    translated_path = output_path / "translated.json"
    translated_path.write_text(str(translated), encoding="utf-8")
    return {"translated": translated}


@app.task(bind=True, max_retries=3, default_retry_delay=2)
def render_task(self, source_path: str, audio_path: str, output_dir: str) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    target = output_path / "final.mp4"
    target.write_bytes(b"mock rendered video")
    return {"output_path": str(target)}
