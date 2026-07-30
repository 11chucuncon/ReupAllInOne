from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import gradio as gr


def _map_voice_label(voice_label: Optional[str]) -> Optional[str]:
    """Convert the UI-friendly voice label to the actual Edge TTS voice identifier."""
    if not voice_label:
        return None
    return {
        "vi-VN-HoaiMyNeural (Nữ Miền Bắc - Truyền cảm)": "vi-VN-HoaiMyNeural",
        "vi-VN-NamMinhNeural (Nam Miền Bắc - Trầm ấm)": "vi-VN-NamMinhNeural",
        "en-US-AndrewMultilingualNeural (Giọng AI Đa ngữ chuẩn)": "en-US-AndrewMultilingualNeural",
        "en-US-AvaMultilingualNeural (Nữ Đa ngữ)": "en-US-AvaMultilingualNeural",
        "en-US-JennyNeural (Giọng nữ tiếng Anh)": "en-US-JennyNeural",
        "en-US-GuyNeural (Giọng nam tiếng Anh)": "en-US-GuyNeural",
        "en-US-ChristopherNeural (Giọng nam rõ ràng)": "en-US-ChristopherNeural",
        "zh-CN-XiaoxiaoNeural (Giọng nữ tiếng Trung)": "zh-CN-XiaoxiaoNeural",
        "ja-JP-NanamiNeural (Giọng nữ tiếng Nhật)": "ja-JP-NanamiNeural",
        "ko-KR-SunHiNeural (Giọng nữ tiếng Hàn)": "ko-KR-SunHiNeural",
        "th-TH-PremwadeeNeural (Giọng nữ tiếng Thái)": "th-TH-PremwadeeNeural",
    }.get(voice_label, voice_label)


def _get_voice_choices(target_language: Optional[str]) -> list[str]:
    """Return TTS voice options filtered by target language."""
    language_map = {
        "vi": [
            "vi-VN-HoaiMyNeural (Nữ Miền Bắc - Truyền cảm)",
            "vi-VN-NamMinhNeural (Nam Miền Bắc - Trầm ấm)",
        ],
        "en": [
            "en-US-JennyNeural (Giọng nữ tiếng Anh)",
            "en-US-GuyNeural (Giọng nam tiếng Anh)",
            "en-US-ChristopherNeural (Giọng nam rõ ràng)",
            "en-US-AndrewMultilingualNeural (Giọng AI Đa ngữ chuẩn)",
            "en-US-AvaMultilingualNeural (Nữ Đa ngữ)",
        ],
        "zh": ["zh-CN-XiaoxiaoNeural (Giọng nữ tiếng Trung)"],
        "ja": ["ja-JP-NanamiNeural (Giọng nữ tiếng Nhật)"],
        "ko": ["ko-KR-SunHiNeural (Giọng nữ tiếng Hàn)"],
        "th": ["th-TH-PremwadeeNeural (Giọng nữ tiếng Thái)"],
    }
    key = (target_language or "vi").split()[0].split("(")[0].strip().lower()
    return language_map.get(key, language_map["vi"])


def _resolve_input_source(uploaded_files: Optional[Sequence[str]], online_links: Optional[str]) -> str:
    """Resolve the first usable input source from uploads or pasted links."""
    if uploaded_files:
        if isinstance(uploaded_files, (list, tuple)):
            for item in uploaded_files:
                if isinstance(item, str) and Path(item).exists():
                    return item
            if uploaded_files:
                return str(uploaded_files[0])
        elif isinstance(uploaded_files, str) and Path(uploaded_files).exists():
            return uploaded_files

    if online_links:
        links = [line.strip() for line in str(online_links).splitlines() if line.strip()]
        if links:
            return links[0]

    raise ValueError("Please provide at least one local video or one valid online link")


def create_app() -> gr.Blocks:
    """Create a modern Gradio UI for the Auto-Reup Video Studio."""
    with gr.Blocks(css="static/style.css", title="AI Video Reup Studio") as demo:
        gr.Markdown("# AI Video Reup Studio")
        gr.Markdown("### Auto-Reup Video with AI Subtitle, TTS, and Render Pipeline")

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["sidebar"]):
                gr.Markdown("## Sidebar")
                gr.Markdown("Quick options and queue status")
                with gr.Tabs():
                    with gr.TabItem("Tải lên từ Máy (Local Upload)"):
                        local_uploads = gr.File(
                            label="Upload video(s)",
                            file_count="multiple",
                            file_types=[".mp4", ".mov", ".mkv"],
                        )
                    with gr.TabItem("Dán Link Online"):
                        online_links = gr.Textbox(
                            label="Paste multiple links (one per line)",
                            lines=4,
                            placeholder="https://www.youtube.com/watch?v=...\nhttps://www.tiktok.com/...",
                        )

                auto_rewrite = gr.Checkbox(label="Rewrite script with AI", value=True)
                target_language = gr.Dropdown(
                    label="Target Language (Ngôn ngữ đích)",
                    choices=[
                        "vi (Tiếng Việt)",
                        "en (Tiếng Anh - English)",
                        "zh (Tiếng Trung - Chinese)",
                        "ja (Tiếng Nhật - Japanese)",
                        "ko (Tiếng Hàn - Korean)",
                        "th (Tiếng Thái - Thai)",
                    ],
                    value="vi (Tiếng Việt)",
                )
                gemini_api_key = gr.Textbox(
                    label="OpenRouter API Key",
                    type="password",
                    placeholder="Enter your OpenRouter API key here",
                )
                tts_engine_mode = gr.Radio(
                    label="TTS Engine",
                    choices=["Edge-TTS Free (Tốc độ cao)", "Local ElevenLabs-Style AI (XTTS v2 Clone)"],
                    value="Edge-TTS Free (Tốc độ cao)",
                )
                reference_audio = gr.Audio(
                    label="Tải lên giọng mẫu (5-10s Audio) để Clone",
                    type="filepath",
                    visible=False,
                )
                voice_preset = gr.Dropdown(
                    label="Preset giọng mẫu",
                    choices=["Nam/Nữ Review phim", "Truyện ngụ ngôn", "News Anchor", "Narrator"],
                    value="Nam/Nữ Review phim",
                    visible=False,
                )
                voice_dropdown = gr.Dropdown(
                    label="TTS Voice",
                    choices=_get_voice_choices("vi"),
                    value="vi-VN-HoaiMyNeural (Nữ Miền Bắc - Truyền cảm)",
                )
                queue_status = gr.Textbox(label="Queue Status", value="Idle", interactive=False)
                submit_btn = gr.Button("Start Reup")

            with gr.Column(scale=2, elem_classes=["main-panel"]):
                gr.Markdown("## Main Workspace")
                video_preview = gr.Video(label="Video Preview")
                output_video = gr.Video(label="Output Video")
                output_text = gr.Textbox(label="Processing Result", lines=6, interactive=False)

            with gr.Column(scale=1, elem_classes=["inspector"]):
                gr.Markdown("## Inspector")
                gr.Markdown("Subtitle and render customization")
                subtitle_font = gr.Dropdown(
                    label="Subtitle Font",
                    choices=["Arial", "Roboto", "Impact", "Montserrat"],
                    value="Arial",
                )
                subtitle_size = gr.Slider(label="Subtitle Size", minimum=12, maximum=72, value=32)
                subtitle_color = gr.ColorPicker(label="Subtitle Color", value="#FFFFFF")
                subtitle_outline_color = gr.ColorPicker(label="Subtitle Border Color", value="#000000")
                subtitle_position = gr.Radio(
                    label="Subtitle Position",
                    choices=["Bottom", "Center"],
                    value="Bottom",
                )
                output_ratio = gr.Radio(
                    label="Output Ratio",
                    choices=["Keep original", "Vertical 9:16 (Shorts/TikTok)", "Horizontal 16:9 (YouTube)"],
                    value="Keep original",
                )
                enable_upscale = gr.Checkbox(
                    label="Bật AI Video Upscale (Nâng cấp độ phân giải)",
                    value=False,
                )
                upscale_factor = gr.Radio(
                    label="Mức độ Upscale",
                    choices=["2x (1080p Full HD)", "4x (4K Ultra HD)"],
                    value="2x (1080p Full HD)",
                )
                speed_slider = gr.Slider(label="Video Speed", minimum=0.8, maximum=1.5, value=1.05, step=0.01)
                hflip_checkbox = gr.Checkbox(label="Flip horizontally", value=True)
                background_audio = gr.File(label="Background Music (optional)", file_types=[".mp3", ".wav", ".m4a"])

        def process_video(
            uploaded_value,
            online_links_value: Optional[str],
            auto_rewrite_value: bool,
            target_language_value: Optional[str],
            openrouter_api_key_value: Optional[str],
            tts_engine_mode_value: Optional[str],
            reference_audio_value,
            voice_preset_value: Optional[str],
            voice_value: Optional[str],
            subtitle_font_value: Optional[str],
            subtitle_size_value: Optional[float],
            subtitle_color_value: Optional[str],
            subtitle_outline_color_value: Optional[str],
            subtitle_position_value: Optional[str],
            output_ratio_value: Optional[str],
            enable_upscale_value: bool,
            upscale_factor_value: Optional[str],
            speed_value: Optional[float],
            hflip_value: Optional[bool],
            background_audio_value,
        ) -> tuple[gr.Video, gr.Video, str, str]:
            try:
                from pipeline import ReupPipeline

                input_source = _resolve_input_source(uploaded_value, online_links_value)
                pipeline = ReupPipeline()
                result_path = pipeline.process_video(
                    input_source=input_source,
                    auto_rewrite=auto_rewrite_value,
                    target_language=target_language_value,
                    custom_voice=_map_voice_label(voice_value),
                    openrouter_api_key=openrouter_api_key_value,
                    tts_engine_mode=tts_engine_mode_value or "Edge-TTS Free (Tốc độ cao)",
                    reference_audio_path=reference_audio_value if isinstance(reference_audio_value, str) else None,
                    voice_preset=voice_preset_value,
                    subtitle_font=subtitle_font_value or "Arial",
                    subtitle_size=int(subtitle_size_value or 32),
                    subtitle_color=subtitle_color_value or "#FFFFFF",
                    subtitle_outline_color=subtitle_outline_color_value or "#000000",
                    subtitle_position=(subtitle_position_value or "Bottom").lower(),
                    output_mode=output_ratio_value or "Keep original",
                    enable_upscale=bool(enable_upscale_value),
                    upscale_factor=upscale_factor_value or "2x (1080p Full HD)",
                    speed_factor=float(speed_value or 1.05),
                    hflip=bool(hflip_value),
                    background_audio_path=background_audio_value[0] if isinstance(background_audio_value, (list, tuple)) and background_audio_value else None,
                )
                return (
                    gr.Video(value=input_source),
                    gr.Video(value=result_path),
                    f"Processed successfully: {result_path}",
                    "Completed",
                )
            except Exception as exc:
                return (
                    gr.Video(value=None),
                    gr.Video(value=None),
                    f"Processing failed: {exc}",
                    "Failed",
                )

        def update_voice_choices(target_language_value: Optional[str]):
            choices = _get_voice_choices(target_language_value)
            return gr.update(choices=choices, value=choices[0] if choices else None)

        def update_tts_controls(tts_engine_mode_value: Optional[str]):
            is_local = str(tts_engine_mode_value or "").startswith("Local")
            return gr.update(visible=is_local), gr.update(visible=is_local)

        target_language.change(
            update_voice_choices,
            inputs=[target_language],
            outputs=[voice_dropdown],
        )
        tts_engine_mode.change(
            update_tts_controls,
            inputs=[tts_engine_mode],
            outputs=[reference_audio, voice_preset],
        )

        submit_btn.click(
            process_video,
            inputs=[
                local_uploads,
                online_links,
                auto_rewrite,
                target_language,
                gemini_api_key,
                tts_engine_mode,
                reference_audio,
                voice_preset,
                voice_dropdown,
                subtitle_font,
                subtitle_size,
                subtitle_color,
                subtitle_outline_color,
                subtitle_position,
                output_ratio,
                enable_upscale,
                upscale_factor,
                speed_slider,
                hflip_checkbox,
                background_audio,
            ],
            outputs=[video_preview, output_video, output_text, queue_status],
        )

    return demo


if __name__ == "__main__":
    demo = create_app()
    demo.launch(share=True, debug=True)
