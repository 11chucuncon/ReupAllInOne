from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import yaml

from app.plugins.asr import ASRStep
from app.plugins.download import DownloadStep
from app.plugins.ffmpeg import FFmpegStep
from app.plugins.ocr import OCRStep
from app.plugins.pipeline import VideoPipeline
from app.plugins.translate import TranslateStep
from app.plugins.translation import TranslationStep
from app.plugins.tts import TTSStep
from app.plugins.watermark import WatermarkStep
from app.plugins.extract_audio import ExtractAudioStep
from app.plugins.subtitles import SubtitleStep


def load_plugin_config(config_path: str | Path = "config_pipeline_full.yaml") -> Dict[str, Any]:
    config_file = Path(config_path)
    if not config_file.exists():
        return {"pipeline": {"steps": [{"name": "DownloadStep", "enabled": True}]}}
    with config_file.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def build_pipeline_from_config(config: Dict[str, Any]) -> VideoPipeline:
    registry = {
        "DownloadStep": DownloadStep(),
        "ExtractAudioStep": ExtractAudioStep(),
        "TranslateStep": TranslateStep(),
        "SubtitleStep": SubtitleStep(),
        "TranslationStep": TranslationStep(),
        "WatermarkStep": WatermarkStep(),
        "ASRStep": ASRStep(),
        "OCRStep": OCRStep(),
        "TTSStep": TTSStep(),
        "FFmpegStep": FFmpegStep(),
    }
    steps_config = config.get("pipeline", {}).get("steps", [])
    enabled_steps = []
    for step in steps_config:
        if not step.get("enabled", True):
            continue
        step_name = step.get("name")
        if step_name in registry:
            plugin = registry[step_name]
            plugin.config = step.get("config", {})
            enabled_steps.append(plugin)
    return VideoPipeline(enabled_steps)


def run_from_config(
    config_path: str | Path = "config_pipeline_full.yaml",
    video_url: str | None = None,
    extra_context: Dict[str, Any] | None = None,
    progress_callback: None | callable = None,
) -> Dict[str, Any]:
    config = load_plugin_config(config_path)
    # allow extra_context to modify pipeline config before building
    if extra_context:
        enable_extract = extra_context.get("enable_extract_audio_step") or extra_context.get("extract_audio")
        if enable_extract:
            steps = config.get("pipeline", {}).get("steps", [])
            found = False
            for step in steps:
                if step.get("name") == "ExtractAudioStep":
                    step["enabled"] = True
                    found = True
                    break
            if not found:
                # insert before ASRStep if present, otherwise append
                insert_idx = 0
                for i, step in enumerate(steps):
                    if step.get("name") == "ASRStep":
                        insert_idx = i
                        break
                steps.insert(insert_idx, {"name": "ExtractAudioStep", "enabled": True, "config": {"output_dir": "outputs"}})
            config.setdefault("pipeline", {})["steps"] = steps

    pipeline = build_pipeline_from_config(config)
    input_value = video_url or "https://example.com/video"
    initial_context: Dict[str, Any] = {"url": input_value, "output_dir": "outputs"}
    if Path(input_value).exists():
        initial_context["video_path"] = input_value
        initial_context["input_path"] = input_value
        initial_context["url"] = None
    if extra_context:
        # merge extra context into initial context (overrides keys)
        initial_context.update(extra_context)
    if progress_callback is not None:
        initial_context["progress_callback"] = progress_callback
    return pipeline.run(initial_context)
