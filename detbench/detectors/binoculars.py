"""Binoculars score = log-PPL(performer) / cross-PPL(observer, performer). Low = AI."""
from __future__ import annotations

import math

from ..text_utils import split_sentences
from .base import DetectionResult, Detector, SentenceScore
from .lm_pair import get_pair, token_mask

# Thresholds published for the falcon-7b / falcon-7b-instruct pair. Any other
# pair must be recalibrated on human text (run.py calibrate does it).
FALCON_LOW_FPR = 0.8536432310785527


class BinocularsDetector(Detector):
    name = "binoculars"
    supports_sentences = True

    def __init__(self, observer: str, performer: str, threshold: float = FALCON_LOW_FPR,
                 quantization: str | None = "8bit", max_tokens: int = 512, temperature: float = 0.05):
        self.pair = get_pair(observer, performer, quantization, max_tokens)
        self.threshold = threshold
        self.temperature = temperature  # softness of the score→probability mapping

    def _prob(self, score: float) -> float:
        return 1 / (1 + math.exp(-(self.threshold - score) / self.temperature))

    def detect(self, text: str) -> DetectionResult:
        st = self.pair.stats(text)
        score = float(st.ce.mean() / st.xent.mean())
        sentences = []
        for sent, a, b in split_sentences(text):
            m = token_mask(st.offsets, a, b)
            if m.sum() >= 3:
                s = float(st.ce[m].mean() / st.xent[m].mean())
                sentences.append(SentenceScore(sent, a, b, self.threshold - s))
        return DetectionResult(self.name, self._prob(score), score < self.threshold,
                               {"score": score, "threshold": self.threshold}, sentences)
