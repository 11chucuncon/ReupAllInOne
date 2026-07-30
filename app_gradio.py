from __future__ import annotations

from pathlib import Path
from typing import Optional

import gradio as gr


def create_app() -> gr.Blocks:
    """Create the Gradio UI for the Auto-Reup Video Studio."""
    with gr.Blocks(css="static/style.css", title="AI Video Reup Studio") as demo:
        gr.Markdown("# AI Video Reup Studio")
        gr.Markdown("### Auto-Reup Video with AI Subtitle, TTS, and Render Pipeline")

        with gr.Row():
            with gr.Column(scale=1, elem_classes=["sidebar"]):
                gr.Markdown("## Sidebar")
                gr.Markdown("Quick options and queue status")
                input_source = gr.Textbox(label="Video URL or Local File", placeholder="https://... or C:/video.mp4")
                auto_rewrite = gr.Checkbox(label="Rewrite script with Gemini", value=True)
                custom_voice = gr.Textbox(label="Custom Voice", placeholder="vi-VN-HoaiMyNeural")
                queue_status = gr.Textbox(label="Queue Status", value="Idle", interactive=False)
                submit_btn = gr.Button("Start Reup")

            with gr.Column(scale=2, elem_classes=["main-panel"]):
                gr.Markdown("## Main Workspace")
                video_preview = gr.Video(label="Video Preview")
                output_video = gr.Video(label="Output Video")
                output_text = gr.Textbox(label="Processing Result", lines=4, interactive=False)

            with gr.Column(scale=1, elem_classes=["inspector"]):
                gr.Markdown("## Inspector")
                gr.Markdown("Subtitle and voice fine tuning")
                subtitle_font = gr.Dropdown(label="Font", choices=["Arial", "Inter", "Roboto"], value="Inter")
                subtitle_size = gr.Slider(label="Font Size", minimum=18, maximum=60, value=32)
                subtitle_color = gr.Dropdown(label="Text Color", choices=["White", "Yellow", "Cyan"], value="White")
                watermark_text = gr.Textbox(label="Watermark", placeholder="AI Reup Studio")

        def process_video(input_source_value: str, auto_rewrite_value: bool, custom_voice_value: Optional[str]) -> tuple[gr.Video, gr.Video, str]:
            try:
                from pipeline import ReupPipeline

                pipeline = ReupPipeline()
                result_path = pipeline.process_video(
                    input_source=input_source_value,
                    auto_rewrite=auto_rewrite_value,
                    custom_voice=custom_voice_value,
                )
                return (
                    gr.Video(value=input_source_value),
                    gr.Video(value=result_path),
                    f"Processed successfully: {result_path}",
                )
            except Exception as exc:
                return (
                    gr.Video(value=None),
                    gr.Video(value=None),
                    f"Processing failed: {exc}",
                )

        submit_btn.click(
            process_video,
            inputs=[input_source, auto_rewrite, custom_voice],
            outputs=[video_preview, output_video, output_text],
        )

    return demo


if __name__ == "__main__":
    demo = create_app()
    demo.launch(share=True, debug=True)
