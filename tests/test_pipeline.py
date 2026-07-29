from pathlib import Path

from app.core.pipeline import build_pipeline_plan, run_pipeline
from app.modules.downloader import VideoDownloader


def test_build_pipeline_plan_returns_expected_steps():
    source_url = "https://example.com/video"
    plan = build_pipeline_plan(source_url)

    assert plan["source_url"] == source_url
    assert [step["name"] for step in plan["steps"]] == [
        "download",
        "audio_separation",
        "transcribe",
        "translate",
        "render",
    ]


def test_downloader_creates_output_file(tmp_path):
    downloader = VideoDownloader()
    output_path = downloader.download("https://example.com/video", tmp_path)

    assert Path(output_path).exists()
    assert Path(output_path).suffix == ".mp4"


def test_run_pipeline_completes(tmp_path):
    job = run_pipeline("https://example.com/video", output_dir=tmp_path)

    assert job["status"] == "completed"
    assert Path(job["output_path"]).exists()
