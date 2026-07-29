from pathlib import Path

from app.workers.orchestrator import CeleryOrchestrator


if __name__ == "__main__":
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    orchestrator = CeleryOrchestrator()
    result = orchestrator.run("https://example.com/video", output_dir=output_dir)
    print(result)
