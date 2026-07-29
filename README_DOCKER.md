# Run the project locally in Docker (Linux-like environment)

This helps emulate a Linux runtime (similar to Colab) so you can test before running on Colab.

Build the Docker image:

```bash
docker build -t video-pipeline:local .
```

Run with Docker:

```bash
docker run --rm -p 7860:7860 -v "%CD%":/app -v "%CD%/downloads":/app/downloads video-pipeline:local
```

Or use docker-compose (recommended):

```bash
docker-compose up --build
```

What it does:
- installs system `ffmpeg` and Python deps
- runs `colab_gradio_ui.py` (it will skip Drive mount if not running on Colab)
- exposes Gradio on http://localhost:7860

Notes:
- If you want to run the pipeline non-interactively, you can run:

```bash
# copy a video into ./downloads
docker run --rm -v "%CD%":/app -v "%CD%/downloads":/app/downloads video-pipeline:local python run_pipeline_cli.py --video downloads/your.mp4
```

- On Linux/macOS, remove the Windows-style `%CD%` and use `$(pwd)` or absolute paths.
