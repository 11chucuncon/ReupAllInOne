# Video Processing Pipeline

## Mục tiêu

Dự án này cung cấp một khung nền để xây dựng một pipeline xử lý video tự động: tải video, tách âm thanh, chuyển văn bản, dịch thuật, và render video cuối cùng.

## Kiến trúc

1. Download video từ nguồn.
2. Tách audio thành vocal và instrumental.
3. Chuyển âm thanh thành văn bản bằng ASR.
4. Dịch văn bản sang ngôn ngữ đích.
5. Render video bằng FFmpeg.

## Kế hoạch triển khai

- Bước 1: Xây dựng cấu trúc module và test đầu tiên.
- Bước 2: Implement pipeline planning.
- Bước 3: Thêm các module downloader, ASR, TTS và renderer.
- Bước 4: Tích hợp CLI và cấu hình.

## Pipeline mở rộng

Mỗi bước có thể được triển khai như một Step riêng biệt hoặc một worker Celery riêng.

### Chạy worker Celery

```bash
celery -A app.workers.tasks worker --loglevel=info -Q video_pipeline
```

### Demo queue end-to-end

```bash
python demo_celery.py
```

### Runner batch thuần Python

```bash
python auto_runner.py
```

### Gọi task mẫu

```python
from app.workers.tasks import download_task

result = download_task.delay("https://example.com/video", "outputs")
print(result.get())
```
