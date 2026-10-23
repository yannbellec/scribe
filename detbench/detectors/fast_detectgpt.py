"""Fast-DetectGPT sampling discrepancy (analytic). High = AI."""
from __future__ import annotations

import math

from ..text_utils import split_sentences
from .base import DetectionResult, Detector, SentenceScore
from .lm_pair import get_pair, token_mask


class FastDetectGPTDetector(Detector):
    name = "fast_detectgpt"
    supports_sentences = True

    def __init__(self, observer: str, performer: str, threshold: float = 1.5,
                 quantization: str | None = "8bit", max_tokens: int = 512, temperature: float = 0.5):
        self.pair = get_pair(observer, performer, quantization, max_tokens)
        self.threshold = threshold  # recalibrate on human text for each model pair
        self.temperature = temperature

    @staticmethod
    def _discrepancy(ll, mean_ref, var_ref) -> float:
        return float((ll.sum() - mean_ref.sum()) / var_ref.sum().clamp_min(1e-6).sqrt())

    def detect(self, text: str) -> DetectionResult:
        st = self.pair.stats(text)
        # Normalise by sqrt(n) so the threshold does not drift with text length.
        n = max(len(st.ll), 1)
        score = self._discrepancy(st.ll, st.mean_ref, st.var_ref) / math.sqrt(n) * math.sqrt(100)
        sentences = []
        for sent, a, b in split_sentences(text):
            m = token_mask(st.offsets, a, b)
            if m.sum() >= 3:
                d = self._discrepancy(st.ll[m], st.mean_ref[m], st.var_ref[m]) / math.sqrt(int(m.sum()))
                sentences.append(SentenceScore(sent, a, b, d))
        prob = 1 / (1 + math.exp(-(score - self.threshold) / self.temperature))
        return DetectionResult(self.name, prob, score > self.threshold,
                               {"score": score, "threshold": self.threshold}, sentences)
