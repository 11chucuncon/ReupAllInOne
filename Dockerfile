FROM python:3.11-slim

# Install system deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg git build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project
COPY . /app

# Install python deps
RUN pip install --no-cache-dir -r requirements.txt gradio PyYAML yt-dlp

EXPOSE 7860

# Run the Gradio UI (colab script will skip Drive mount when not in Colab)
CMD ["python", "colab_gradio_ui.py"]
