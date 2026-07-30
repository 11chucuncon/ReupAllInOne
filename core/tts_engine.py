from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Optional

import edge_tts
import yaml

logger = logging.getLogger(__name__)


class TTSEngine:
    """Generate speech audio using Edge-TTS or a local XTTS v2 voice-cloning backend."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        self.config_path = config_path or str(Path(__file__).resolve().parents[1] / "config" / "settings.yaml")
        self.settings = self._load_settings()
        self.tts_config = self.settings.get("tts", {})
        self.default_voice = self.tts_config.get("default_voice", "vi-VN-HoaiMyNeural")
        self.default_rate = self.tts_config.get("rate", "+0%")
        self.default_volume = self.tts_config.get("volume", "+0%")
        self.project_root = Path(__file__).resolve().parents[1]
        self.xtts_model_name = self.tts_config.get("xtts_model_name", "tts_models/multilingual/multi-dataset/xtts_v2")
        self._xtts_model = None

    def _load_settings(self) -> dict:
        """Load project settings from config/settings.yaml."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as handle:
                return yaml.safe_load(handle) or {}
        except FileNotFoundError as exc:
            logger.error("Settings file not found at %s", self.config_path)
            raise FileNotFoundError(f"Missing configuration file: {self.config_path}") from exc
        except yaml.YAMLError as exc:
            logger.error("Failed to parse settings YAML: %s", exc)
            raise RuntimeError("Invalid YAML configuration") from exc

    def _normalize_language(self, target_language: Optional[str]) -> str:
        """Normalize a target language code for XTTS."""
        language_code = (target_language or "vi").split()[0].split("(")[0].strip().lower()
        return {
            "vi": "vi-VN",
            "en": "en-US",
            "zh": "zh-CN",
            "ja": "ja-JP",
            "ko": "ko-KR",
            "th": "th-TH",
        }.get(language_code, "en-US")

    def _resolve_reference_audio(self, reference_audio_path: Optional[str], voice_preset: Optional[str] = None) -> Optional[Path]:
        """Resolve a local reference audio file for voice cloning."""
        candidates: list[Path] = []
        if reference_audio_path:
            candidates.append(Path(reference_audio_path).expanduser())

        if voice_preset:
            preset_name = "_".join(str(voice_preset).lower().replace("/", "_").split())
            for suffix in (".wav", ".mp3", ".m4a", ".ogg"):
                candidates.append(self.project_root / "assets" / "voices" / f"{preset_name}{suffix}")

        assets_dir = self.project_root / "assets" / "voices"
        if assets_dir.exists():
            for suffix in (".wav", ".mp3", ".m4a", ".ogg"):
                for match in sorted(assets_dir.glob(f"*{suffix}")):
                    candidates.append(match)

        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return None

    def _ensure_xtts_model(self):
        """Load the XTTS v2 model lazily on first use."""
        if self._xtts_model is not None:
            return self._xtts_model

        try:
            from TTS.api import TTS as CoquiTTS
        except Exception as exc:  # pragma: no cover - depends on optional runtime deps
            raise RuntimeError(f"XTTS dependencies are not installed: {exc}") from exc

        try:
            import torch
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"PyTorch is not installed: {exc}") from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading XTTS v2 on %s", device)
        self._xtts_model = CoquiTTS(self.xtts_model_name, progress_bar=False, gpu=device == "cuda")
        return self._xtts_model

    def _get_audio_duration(self, audio_path: Path) -> float:
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
                capture_output=True,
                text=True,
                check=True,
            )
            return float(result.stdout.strip() or 0.0)
        except Exception as exc:
            logger.warning("Could not inspect audio duration for %s: %s", audio_path, exc)
            return 0.0

    def _align_audio_to_reference(self, output_file: Path, target_duration: float) -> Path:
        if target_duration <= 0:
            return output_file

        actual_duration = self._get_audio_duration(output_file)
        if actual_duration <= 0:
            return output_file

        ratio = target_duration / actual_duration
        if ratio >= 0.95 and ratio <= 1.05:
            return output_file

        if ratio < 1.0:
            padded_output = output_file.with_suffix(".padded.mp3")
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(output_file),
                    "-af",
                    f"apad=whole_dur={target_duration:.3f}",
                    str(padded_output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            return padded_output

        adjusted_output = output_file.with_suffix(".adjusted.mp3")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(output_file),
                "-filter:a",
                f"atempo={ratio:.3f}",
                str(adjusted_output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return adjusted_output

    async def _generate_edge_speech(self, text: str, output_file: Path, voice: Optional[str], rate: Optional[str]) -> str:
        """Generate speech audio using Edge-TTS."""
        selected_voice = voice or self.default_voice
        selected_rate = rate or self.default_rate
        communicate = edge_tts.Communicate(text, voice=selected_voice, rate=selected_rate)
        await communicate.save(str(output_file))
        logger.info("Generated speech audio at %s", output_file)
        return str(output_file)

    async def clone_speech(
        self,
        text: str,
        output_path: str,
        reference_audio_path: Optional[str] = None,
        target_language: Optional[str] = None,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        voice_preset: Optional[str] = None,
        target_duration: Optional[float] = None,
    ) -> str:
        """Clone a voice using a local XTTS v2 model with a reference audio sample."""
        if not text or not text.strip():
            raise ValueError("Text input cannot be empty")

        output_file = Path(output_path)
        if output_file.suffix.lower() != ".mp3":
            output_file = output_file.with_suffix(".mp3")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        reference_audio = self._resolve_reference_audio(reference_audio_path, voice_preset=voice_preset)
        if reference_audio is None:
            logger.warning("No reference audio found for voice cloning; falling back to Edge-TTS")
            generated_path = await self._generate_edge_speech(text, output_file, voice, rate)
            return str(self._align_audio_to_reference(Path(generated_path), target_duration or 0.0))

        try:
            tts_model = self._ensure_xtts_model()
            language_code = self._normalize_language(target_language)
            logger.info("Cloning voice from %s using XTTS v2", reference_audio)
            await asyncio.to_thread(
                tts_model.tts_to_file,
                text,
                str(output_file),
                speaker_wav=str(reference_audio),
                language=language_code,
            )
            logger.info("Generated cloned speech audio at %s", output_file)
            return str(self._align_audio_to_reference(output_file, target_duration or 0.0))
        except Exception as exc:
            logger.warning("XTTS voice cloning failed: %s; falling back to Edge-TTS", exc)
            generated_path = await self._generate_edge_speech(text, output_file, voice, rate)
            return str(self._align_audio_to_reference(Path(generated_path), target_duration or 0.0))

    async def generate_speech(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        rate: Optional[str] = None,
        engine_mode: str = "edge",
        reference_audio_path: Optional[str] = None,
        target_language: Optional[str] = None,
        voice_preset: Optional[str] = None,
        target_duration: Optional[float] = None,
    ) -> str:
        """Generate an MP3 audio file using the requested engine."""
        if not text or not text.strip():
            raise ValueError("Text input cannot be empty")

        output_file = Path(output_path)
        if output_file.suffix.lower() != ".mp3":
            output_file = output_file.with_suffix(".mp3")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if str(engine_mode or "").lower().startswith("local"):
            generated_path = await self.clone_speech(
                text,
                str(output_file),
                reference_audio_path=reference_audio_path,
                target_language=target_language,
                voice=voice,
                rate=rate,
                voice_preset=voice_preset,
            )
            return str(self._align_audio_to_reference(Path(generated_path), target_duration or 0.0))

        generated_path = await self._generate_edge_speech(text, output_file, voice, rate)
        return str(self._align_audio_to_reference(Path(generated_path), target_duration or 0.0))
