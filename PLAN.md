# Kế hoạch triển khai cụ thể

## Mục tiêu MVP
Xây dựng một pipeline xử lý video theo từng bước rõ ràng:
1. Nhận URL video đầu vào.
2. Tải video về.
3. Tách âm thanh thành vocal/instrumental.
4. Chuyển âm thanh thành văn bản.
5. Dịch và chuẩn hóa lời thoại.
6. Render video đầu ra.

## Phân chia công việc

### Giai đoạn 1: Nền tảng dự án
- Tạo cấu trúc thư mục:
  - app/core/
  - app/modules/
  - app/utils/
  - tests/
- Thêm file cấu hình cơ bản: requirements.txt, README.md, .gitignore.
- Xây dựng lớp pipeline orchestration để chạy các bước tuần tự.

### Giai đoạn 2: Core pipeline
- Tạo file app/core/pipeline.py để định nghĩa luồng chạy.
- Tạo class PipelineJob lưu trạng thái: pending, running, completed, failed.
- Tạo hàm build_pipeline_plan(source_url) trả về danh sách các bước.
- Thêm test đầu tiên cho việc tạo plan.

### Giai đoạn 3: Module download
- Tạo file app/modules/downloader.py.
- Dùng yt-dlp để download video từ URL.
- Lưu file vào thư mục outputs/<job_id>/source.mp4.
- Xử lý lỗi khi URL không hợp lệ hoặc network bị chặn.

### Giai đoạn 4: Module xử lý audio
- Tạo file app/modules/audio_processor.py.
- Tách audio thành 2 luồng: vocal và instrumental.
- Nếu thư viện tách âm thanh chưa sẵn sàng, dùng placeholder để giữ cấu trúc code chuẩn.
- Mục tiêu sau này có thể tích hợp UVR/MDX-Net.

### Giai đoạn 5: Module ASR
- Tạo file app/modules/asr.py.
- Chuyển vocal thành transcript theo định dạng:
  - start_time
  - end_time
  - text
- Dự phòng dùng WhisperX hoặc FunASR khi môi trường hỗ trợ.

### Giai đoạn 6: Module translate và subtitle
- Tạo file app/modules/translator.py.
- Dịch transcript sang ngôn ngữ đích.
- Chuẩn hóa chuỗi để phù hợp với TTS và thời lượng video.
- Tạo cấu trúc JSON cho subtitle.

### Giai đoạn 7: Module TTS và audio mixing
- Tạo file app/modules/tts.py.
- Tạo âm thanh đọc lời thoại bằng edge-tts hoặc XTTS-v2.
- Ghép audio tts + instrumental thành file audio cuối.

### Giai đoạn 8: Module render video
- Tạo file app/modules/renderer.py.
- Dùng FFmpeg để render video finale.
- Hỗ trợ thêm: che logo cũ, chèn watermark, và mix audio.

### Giai đoạn 9: CLI và vận hành
- Mở rộng app/cli.py để nhận tham số:
  - --source-url
  - --output-dir
  - --language
- In ra log và trạng thái từng bước.

## Thứ tự ưu tiên triển khai
1. pipeline.py
2. downloader.py
3. audio_processor.py
4. asr.py
5. translator.py
6. tts.py
7. renderer.py
8. CLI và logging

## Yêu cầu test
- Test cho plan generation.
- Test cho downloader với URL giả.
- Test cho parser transcript.
- Test cho renderer command build.

## Kết quả mong đợi sau giai đoạn đầu
- Có thể chạy một job từ URL -> tạo thư mục outputs -> output video mẫu.
- Mỗi module có thể hoạt động độc lập và dễ nối vào pipeline.
