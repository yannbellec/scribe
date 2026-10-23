"""Pangram Labs API (https://pangram.readthedocs.io/en/stable/api/rest.html).

The free tier is ~2,000 words/day, so results are cached on disk by text hash
and a daily word budget is enforced before any call.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import time
from pathlib import Path

import requests

from ..text_utils import word_count
from .base import DetectionResult, Detector, SentenceScore

API = "https://text.external-api.pangram.com"


class PangramDetector(Detector):
    name = "pangram"
    supports_sentences = True

    def __init__(self, api_key: str | None = None, cache_dir: str = "results/cache/pangram",
                 daily_word_budget: int = 2000, timeout: int = 300):
        self.api_key = api_key or os.environ.get("PANGRAM_API_KEY")
        if not self.api_key:
            raise RuntimeError("PANGRAM_API_KEY manquant (voir .env.example)")
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.budget = daily_word_budget
        self.timeout = timeout

    def _usage_file(self) -> Path:
        return self.cache / f"usage-{dt.date.today().isoformat()}.json"

    def _words_used_today(self) -> int:
        f = self._usage_file()
        return json.loads(f.read_text())["words"] if f.exists() else 0

    def _record_usage(self, words: int) -> None:
        self._usage_file().write_text(json.dumps({"words": self._words_used_today() + words}))

    def _call(self, text: str) -> dict:
        headers = {"x-api-key": self.api_key}
        r = requests.post(f"{API}/task", headers=headers, json={"text": text}, timeout=60)
        r.raise_for_status()
        task_id = r.json()["task_id"]
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            g = requests.get(f"{API}/task/{task_id}", headers=headers, timeout=60)
            g.raise_for_status()
            data = g.json()
            if data.get("stage") == "STAGE_SUCCESS":
                return data
            if data.get("stage") == "STAGE_FAILED":
                raise RuntimeError(f"Pangram: {data.get('headline')}")
            time.sleep(1)
        raise TimeoutError("Pangram: délai dépassé")

    def detect(self, text: str) -> DetectionResult:
        key = hashlib.sha256(text.encode()).hexdigest()[:24]
        cached = self.cache / f"{key}.json"
        if cached.exists():
            data = json.loads(cached.read_text())
        else:
            words = word_count(text)
            if self._words_used_today() + words > self.budget:
                raise RuntimeError(
                    f"Budget Pangram du jour atteint ({self.budget} mots). Relancer demain "
                    "ou augmenter pangram.daily_word_budget dans config.yaml.")
            data = self._call(text)
            self._record_usage(words)
            cached.write_text(json.dumps(data, ensure_ascii=False))

        p_ai = float(data.get("fraction_ai", 0)) + 0.5 * float(data.get("fraction_ai_assisted", 0))
        sentences = [
            SentenceScore(w.get("text", ""), int(w.get("start_index", 0)), int(w.get("end_index", 0)),
                          float(w.get("ai_assistance_score", 0)))
            for w in data.get("windows", [])
        ]
        return DetectionResult(
            detector=self.name,
            ai_probability=min(p_ai, 1.0),
            flagged=data.get("prediction_short", "").lower() not in ("human", ""),
            raw={k: data.get(k) for k in ("version", "headline", "prediction_short", "fraction_ai",
                                           "fraction_ai_assisted", "fraction_human")},
            sentences=sentences,
        )
