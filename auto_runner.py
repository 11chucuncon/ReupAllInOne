import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

STATUS_FILE = "queue_status.json"


def process_single_video(video_url: str) -> Dict[str, Any]:
    print(f"🚀 Bắt đầu xử lý: {video_url}")
    time.sleep(3)
    return {"url": video_url, "status": "SUCCESS", "error": None}


def load_status() -> Dict[str, Any]:
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_status(status_data: Dict[str, Any]) -> None:
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(status_data, f, ensure_ascii=False, indent=2)


def run_batch_processor(video_links: List[str], max_workers: int = 2) -> None:
    status_data = load_status()
    pending_links = [link for link in video_links if status_data.get(link, {}).get("status") != "SUCCESS"]

    print(f"📌 Tổng link: {len(video_links)} | Cần xử lý: {len(pending_links)}")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_video, link): link for link in pending_links}

        for future in as_completed(futures):
            link = futures[future]
            try:
                result = future.result()
                status_data[link] = result
                print(f"✅ Hoàn thành: {link}")
            except Exception as exc:
                status_data[link] = {"url": link, "status": "FAILED", "error": str(exc)}
                print(f"❌ Lỗi khi làm video {link}: {exc}")

            save_status(status_data)


if __name__ == "__main__":
    my_links = [
        "https://www.douyin.com/video/1111111",
        "https://www.douyin.com/video/2222222",
        "https://www.douyin.com/video/3333333",
    ]
    run_batch_processor(my_links, max_workers=2)
