"""Common interface for every detector in the benchmark."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SentenceScore:
    text: str
    start: int
    end: int
    ai_score: float  # higher = more "AI-like", detector-specific scale


@dataclass
class DetectionResult:
    detector: str
    ai_probability: float  # 0..1, comparable across detectors
    flagged: bool  # verdict at the detector's own operating threshold
    raw: dict = field(default_factory=dict)
    sentences: list[SentenceScore] = field(default_factory=list)


class Detector:
    name: str = "base"
    # True when sentence-level scores are available (used by the revision loop)
    supports_sentences: bool = False

    def detect(self, text: str) -> DetectionResult:
        raise NotImplementedError
