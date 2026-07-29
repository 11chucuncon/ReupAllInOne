# Chạy backend trên Google Colab

Mở `run_on_colab.ipynb` trong Google Colab để chạy backend và UI trên máy ảo của Colab.

## Các bước

1. Mở notebook `run_on_colab.ipynb`.
2. Chọn `Runtime -> Run all`.
3. Chờ quá trình cài đặt và clone hoàn tất.
4. Khi cell cuối cùng chạy, nó sẽ khởi động Gradio UI và in ra đường dẫn public.

## Lưu ý

- Notebook sử dụng `share=True` nên sẽ tạo Gradio public URL tạm thời.
- Nếu muốn lưu output vào Drive, dùng `outputs/` trong repo và Drive đã mounted.
- Để chạy lại, bạn có thể dừng notebook và chạy lại cell cuối.
