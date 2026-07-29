import os

from app.plugins.subtitles import SubtitleStep


def test_subtitle_step_writes_srt(tmp_path):
    # prepare fake ASR result with timestamps
    audio_file = tmp_path / "audio.wav"
    audio_file.write_text("")

    timestamps = [
        {"start": 0.0, "end": 2.0, "text": "Hello world"},
        {"start": 2.5, "end": 5.0, "text": "This is a test"},
    ]

    context = {
        "asr_result": {
            "audio_path": str(audio_file),
            "timestamps": timestamps,
        },
        "translation_target": None,
    }

    step = SubtitleStep()
    out = step.execute(context)
    assert out["subtitle_result"]["status"] == "written"
    srt_path = out["subtitle_result"]["srt_path"]
    assert os.path.exists(srt_path)
    content = open(srt_path, "r", encoding="utf-8").read()
    assert "Hello world" in content
    assert "This is a test" in content
