from __future__ import annotations

from typing import List, Dict


class Translator:
    def translate(self, transcript: List[Dict[str, str]]) -> List[Dict[str, str]]:
        translated = []
        for item in transcript:
            translated.append(
                {
                    "start": item["start"],
                    "end": item["end"],
                    "text": f"Translated: {item['text']}",
                }
            )
        return translated
