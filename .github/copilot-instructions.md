# Copilot Instructions for AI Video Reup Studio

## 1. Project Overview

Project Name: AI Video Reup Studio

This project is an Auto-Reup Video system that runs backend processing on Google Colab and exposes a public web interface via Gradio.

The system must be designed using a Modular Architecture with clear separation of concerns so that it is:
- Easy to fix
- Easy to extend with new features
- Compatible with multi-threading / concurrent execution using ThreadPoolExecutor

## 2. Technology Stack

The project must use only free and open-source technologies:
- Downloader: yt-dlp
- ASR / Subtitle + Timestamp generation: faster-whisper (CUDA GPU)
- LLM for script rewriting: google-generativeai (Gemini 1.5 Flash API)
- TTS for AI voice: edge-tts (Microsoft Edge Free)
- Media rendering: FFmpeg CLI (including hflip, speed adjustment, subtitle embedding, NVENC GPU)
- UI / Web: gradio + pyngrok

## 3. Coding Standards

### General Rules
- Language: Python 3.10+
- Follow Clean Code principles
- All classes and functions must include Type Hinting
- All classes and functions must include Docstrings
- Avoid hardcoded values for:
  - API keys
  - FFmpeg parameters
  - prompts
  - voice settings
  - other configuration values
- All configuration must be loaded from config/settings.yaml

### Architecture Rules
- Each file inside core/ must follow the Single Responsibility Principle (SRP)
- Each file must do exactly one job and one job only
- Keep modules loosely coupled and easy to maintain

### Error Handling and Logging
- Write complete try/except exception handling
- Use the logging module for clear and structured logs
- Do not silently swallow exceptions; log them properly

## 4. UI Frontend Guidelines (Gradio Layout & Styling)

- Use Gradio via gr.Blocks with a custom dark theme style.
- The visual design should follow a modern dark theme with:
  - primary background: #0b0f19
  - accent green: #00c853
  - accent pink: #ff4081
- The interface should be structured as a 3-column dashboard layout:
  - Left Column (Sidebar): Quick options, SRT/Music import, Queue Status
  - Middle Column (Main Area): Video input/link, video preview player, output format selection (Landscape/Portrait), speed control, primary action button
  - Right Column (Inspector): Subtitle fine-tuning controls such as font, size, color, border, shadow, text overlay, and watermark
- All styling must be defined in static/style.css and imported into app_gradio.py.
- Avoid excessive inline CSS inside Python code; keep UI styling centralized and maintainable.

## 5. AI Workspace Guidance

When working on this project, AI assistants must:
- Respect the modular architecture
- Keep changes scoped and maintainable
- Prefer configuration-driven implementation over hardcoding
- Preserve separation of concerns between downloader, ASR, LLM, TTS, rendering, and UI layers
- Ensure new features can be added without tightly coupling components
- Keep concurrency support in mind when adding new processing steps

## 6. Implementation Expectations

- Prefer readable, maintainable, and testable code
- Keep business logic separate from UI logic
- Keep processing logic separate from configuration logic
- Ensure compatibility with Colab-based execution environments where appropriate
- Favor explicit, well-documented code over clever but opaque implementation
