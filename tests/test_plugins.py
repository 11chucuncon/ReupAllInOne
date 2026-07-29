import json
from pathlib import Path

import app.plugins.asr as asr_module
import app.plugins.ffmpeg as ffmpeg_module
import app.plugins.ocr as ocr_module
import app.plugins.translation as translation_module
import app.plugins.tts as tts_module
from app.plugins.asr import ASRStep
from app.plugins.ffmpeg import FFmpegStep
from app.plugins.ocr import OCRStep
from app.plugins.runner import build_pipeline_from_config, run_from_config
from app.plugins.translation import TranslationStep
from app.plugins.tts import TTSStep


def test_build_pipeline_from_config_uses_enabled_steps():
    pipeline = build_pipeline_from_config({"pipeline": {"steps": [{"name": "DownloadStep", "enabled": True}, {"name": "TranslateStep", "enabled": True}]}})
    assert len(pipeline.steps) == 2


def test_run_from_config_returns_context():
    result = run_from_config("config_pipeline.yaml")
    assert result["video_path"].replace("\\", "/") == "downloads/raw.mp4"
    assert result["translated_text"] == "Translated text"
    assert result["rendered_path"].replace("\\", "/") == "outputs/final.mp4"


def test_translation_step_uses_openrouter_when_api_key_present(monkeypatch):
    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "Bonjour"}}]}).encode("utf-8")

    def fake_urlopen(request, timeout):
        assert request.full_url.startswith("https://openrouter.ai/api/v1/chat/completions")
        assert "Authorization" in dict(request.header_items())
        return DummyResponse()

    monkeypatch.setattr(translation_module.request, "urlopen", fake_urlopen)

    step = TranslationStep(config={"provider": "openrouter", "api_key": "test-key", "model": "deepseek/deepseek-v4-flash"})
    result = step.execute({"asr_result": {"text": "Hello"}})

    assert result["translation_result"]["status"] == "translated"
    assert result["translation_result"]["translated_text"] == "Bonjour"


def test_translation_step_resolves_env_placeholders_in_config(monkeypatch):
    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "Salut"}}]}).encode("utf-8")

    monkeypatch.setenv("OPENROUTER_API_KEY", "placeholder-key")
    monkeypatch.setattr(translation_module.request, "urlopen", lambda request, timeout: DummyResponse())

    step = TranslationStep(config={"provider": "openrouter", "api_key": "${OPENROUTER_API_KEY}", "model": "deepseek/deepseek-v4-flash"})
    result = step.execute({"asr_result": {"text": "Hello"}})

    assert result["translation_result"]["status"] == "translated"
    assert result["translation_result"]["translated_text"] == "Salut"


def test_load_env_file_overrides_existing_environment(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "stale-key")
    step = TranslationStep(config={"provider": "openrouter"})

    step._load_env_file()

    env_value = translation_module.os.getenv("OPENROUTER_API_KEY", "")
    assert env_value.startswith("sk-or-v1-")
    assert env_value != "stale-key"


def test_ffmpeg_step_falls_back_to_mp4_webm_input(tmp_path, monkeypatch):
    downloads_dir = tmp_path / "downloads"
    downloads_dir.mkdir()
    input_path = downloads_dir / "raw.mp4.webm"
    input_path.write_bytes(b"fake")

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    def fake_run(cmd, check=True, stdout=None, stderr=None, text=None, cwd=None):
        target_dir = Path(cwd) if cwd else Path.cwd()
        output_path = target_dir / Path(cmd[-1]).name
        output_path.write_bytes(b"encoded")
        return type("CompletedProcess", (), {"stdout": "", "returncode": 0})()

    monkeypatch.setattr(ffmpeg_module.shutil, "which", lambda _: "ffmpeg")
    monkeypatch.setattr(ffmpeg_module.subprocess, "run", fake_run)

    step = FFmpegStep()
    result = step.execute({"video_path": str(downloads_dir / "raw.mp4"), "output_dir": str(output_dir)})

    assert result["render_status"] == "success"
    assert (output_dir / "final.mp4").exists()


def test_ffmpeg_step_writes_subtitle_file_when_caption_data_exists(tmp_path, monkeypatch):
    downloads_dir = tmp_path / "downloads"
    downloads_dir.mkdir()
    input_path = downloads_dir / "raw.mp4"
    input_path.write_bytes(b"fake")

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    seen_cmd = {}

    def fake_run(cmd, check=True, stdout=None, stderr=None, text=None, cwd=None):
        seen_cmd["cmd"] = cmd
        seen_cmd["cwd"] = cwd
        target_dir = Path(cwd) if cwd else Path.cwd()
        output_path = target_dir / Path(cmd[-1]).name
        output_path.write_bytes(b"encoded")
        return type("CompletedProcess", (), {"stdout": "", "returncode": 0})()

    monkeypatch.setattr(ffmpeg_module.shutil, "which", lambda _: "ffmpeg")
    monkeypatch.setattr(ffmpeg_module.subprocess, "run", fake_run)

    step = FFmpegStep()
    result = step.execute({
        "video_path": str(input_path),
        "output_dir": str(output_dir),
        "translation_result": {"translated_text": "Xin chào"},
        "asr_result": {"timestamps": [{"start": 0.0, "end": 2.0, "text": "hello"}]},
    })

    assert result["render_status"] == "success"
    assert (output_dir / "subtitles.srt").exists()
    assert any("subtitles=" in arg for arg in seen_cmd["cmd"])
    assert seen_cmd["cwd"] == str(output_dir)
    assert seen_cmd["cwd"] == str(output_dir)


def test_ffmpeg_step_can_use_source_or_translated_subtitles(tmp_path, monkeypatch):
    downloads_dir = tmp_path / "downloads"
    downloads_dir.mkdir()
    input_path = downloads_dir / "raw.mp4"
    input_path.write_bytes(b"fake")

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    monkeypatch.setattr(ffmpeg_module.shutil, "which", lambda _: "ffmpeg")
    monkeypatch.setattr(ffmpeg_module.subprocess, "run", lambda cmd, check=True, stdout=None, stderr=None, text=None, cwd=None: type("CompletedProcess", (), {"stdout": "", "returncode": 0})())

    step = FFmpegStep()
    result = step.execute({
        "video_path": str(input_path),
        "output_dir": str(output_dir),
        "subtitle_mode": "source",
        "translation_result": {"translated_text": "Xin chào"},
        "asr_result": {"timestamps": [{"start": 0.0, "end": 2.0, "text": "hello"}]},
    })

    assert (output_dir / "subtitles.srt").exists()
    assert "hello" in (output_dir / "subtitles.srt").read_text(encoding="utf-8")


def test_ffmpeg_step_falls_back_to_common_locations_when_path_missing(tmp_path, monkeypatch):
    downloads_dir = tmp_path / "downloads"
    downloads_dir.mkdir()
    input_path = downloads_dir / "raw.mp4"
    input_path.write_bytes(b"fake")

    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    ffmpeg_path = tmp_path / "ffmpeg"
    ffmpeg_path.write_bytes(b"#!/bin/sh\nexit 0\n")
    ffmpeg_path.chmod(0o755)

    seen_cmd = {}

    def fake_run(cmd, check=True, stdout=None, stderr=None, text=None, cwd=None):
        seen_cmd["cmd"] = cmd
        seen_cmd["cwd"] = cwd
        target_dir = Path(cwd) if cwd else Path.cwd()
        output_path = target_dir / Path(cmd[-1]).name
        output_path.write_bytes(b"encoded")
        return type("CompletedProcess", (), {"stdout": "", "returncode": 0})()

    monkeypatch.setattr(ffmpeg_module.shutil, "which", lambda _: None)
    monkeypatch.setattr(ffmpeg_module.subprocess, "run", fake_run)
    monkeypatch.setattr(ffmpeg_module, "_candidate_ffmpeg_paths", lambda: [str(ffmpeg_path)])

    step = FFmpegStep()
    result = step.execute({"video_path": str(input_path), "output_dir": str(output_dir)})

    assert result["render_status"] == "success"
    assert seen_cmd["cmd"][0] == str(ffmpeg_path)


def test_asr_step_uses_backend_when_available(tmp_path, monkeypatch):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake")

    monkeypatch.setattr(asr_module, "_transcribe_with_faster_whisper", lambda self, audio_file, language: ("hello world", [{"start": 0.0, "end": 1.0, "text": "hello world"}], {"language": "en"}))

    step = ASRStep(config={"engine": "faster_whisper", "language": "en"})
    result = step.execute({"video_path": str(audio_path)})

    assert result["asr_result"]["status"] == "transcribed"
    assert result["asr_result"]["text"] == "hello world"


def test_ocr_step_uses_pytesseract_when_available(tmp_path, monkeypatch):
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"fake")

    monkeypatch.setattr(ocr_module, "_ocr_with_pytesseract", lambda self, image_file, language: "hello")

    step = OCRStep(config={"engine": "pytesseract", "language": "eng"})
    result = step.execute({"video_path": str(image_path)})

    assert result["ocr_result"]["status"] == "recognized"
    assert result["ocr_result"]["text"] == "hello"


def test_tts_step_writes_audio_when_backend_available(tmp_path, monkeypatch):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    monkeypatch.setattr(tts_module, "_speak_to_file_with_pyttsx3", lambda self, output_path, text: output_path.write_bytes(b"audio"))

    step = TTSStep(config={"engine": "pyttsx3", "output_dir": str(output_dir)})
    result = step.execute({"translation_result": {"translated_text": "hola"}})

    assert result["tts_result"]["status"] == "generated"
    assert (output_dir / "tts.wav").exists()
