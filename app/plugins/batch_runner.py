from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from app.plugins.runner import run_from_config


def run_batch_from_config(config_path: str | Path, video_inputs: Iterable[str | Path], output_root: str | Path = "outputs/batch") -> List[Dict[str, Any]]:
    config = Path(config_path)
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)

    results = []
    for index, video_input in enumerate(video_inputs, start=1):
        item_output_dir = output_root_path / f"job_{index:03d}"
        item_output_dir.mkdir(parents=True, exist_ok=True)
        result = run_from_config(config, video_url=str(video_input))
        result["job_output_dir"] = str(item_output_dir)
        result["job_index"] = index
        results.append(result)

    return results
