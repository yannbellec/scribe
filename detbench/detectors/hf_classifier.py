"""Supervised Hugging Face classifier (RoBERTa / DeBERTa style). English-centric."""
from __future__ import annotations

import torch

from .base import DetectionResult, Detector


class HFClassifierDetector(Detector):
    name = "hf_classifier"

    def __init__(self, model: str, ai_label: str | int | None = None, threshold: float = 0.5,
                 max_tokens: int = 512):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).eval()
        if torch.cuda.is_available():
            self.model.to("cuda")
        self.name = f"hf:{model.split('/')[-1]}"
        self.threshold = threshold
        self.max_tokens = max_tokens
        self.ai_index = self._resolve_label(ai_label)

    def _resolve_label(self, ai_label) -> int:
        if isinstance(ai_label, int):
            return ai_label
        id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        wanted = [str(ai_label).lower()] if ai_label else ["ai", "fake", "machine", "generated", "label_1"]
        for idx, label in id2label.items():
            if any(w in label for w in wanted):
                return idx
        return max(id2label)  # convention: last class = AI

    @torch.inference_mode()
    def detect(self, text: str) -> DetectionResult:
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_tokens)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        logits = self.model(**enc).logits[0]
        if logits.numel() == 1:
            p = float(torch.sigmoid(logits)[0])
        else:
            p = float(torch.softmax(logits, -1)[self.ai_index])
        return DetectionResult(self.name, p, p >= self.threshold, {"p_ai": p})
