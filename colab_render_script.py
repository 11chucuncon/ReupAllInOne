import os
import sys
import shutil
from pathlib import Path

repo = Path('/content/video-pipeline')
if not repo.exists():
    raise SystemExit('Clone or upload the project to /content/video-pipeline first')

os.chdir(repo)
sys.path.insert(0, str(repo))

if shutil.which('ffmpeg') is None:
    os.system('apt-get update -y')
    os.system('apt-get install -y ffmpeg')

from app.plugins.runner import run_from_config

video_input = '/content/drive/MyDrive/video.mp4'
result = run_from_config('config_pipeline_full.yaml', video_url=video_input)
print(result)
