"""Open Pangram / EditLens (Pangram Labs, ICLR 2026), github.com/pangramlabs/EditLens.

Open-weight version of Pangram: a sequence classifier over buckets of "amount of
AI editing"; the score is the expected bucket position in [0, 1]. Weights are
CC BY-NC-SA 4.0 (non-commercial research use) and gated on Hugging Face, as is
the Llama base: accept both licences and run `huggingface-cli login` first.

`mirror` loads the adapter from a public re-upload instead (the licence allows
non-commercial redistribution). The mirror is only used if its adapter file has
exactly the SHA-256 that the Hub publishes for the original Pangram file, which
is readable even while access to the original is pending.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from ..device import best_device
from .base import DetectionResult, Detector

BASES = {
    "pangram/editlens_Llama-3.2-3B": ("meta-llama/Llama-3.2-3B", 1024),
    "pangram/editlens_roberta-large": ("FacebookAI/roberta-large", 512),
}


ADAPTER_FILE = "adapter_model.safetensors"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def original_sha256(checkpoint: str) -> str:
    """SHA-256 the Hub publishes for the original adapter (file metadata is public on gated repos)."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(checkpoint, files_metadata=True)
    for sib in info.siblings:
        if sib.rfilename == ADAPTER_FILE and sib.lfs:
            return sib.lfs.sha256
    raise RuntimeError(f"Empreinte de {checkpoint}/{ADAPTER_FILE} introuvable sur le Hub")


def verify_mirror(checkpoint: str, adapter_dir: str, expected: str | None = None) -> None:
    expected = expected or original_sha256(checkpoint)
    actual = sha256_file(Path(adapter_dir) / ADAPTER_FILE)
    if actual != expected:
        raise RuntimeError(f"La copie ne correspond pas à l'original {checkpoint} "
                           f"(SHA-256 {actual[:12]}… au lieu de {expected[:12]}…) : chargement refusé")


class EditLensDetector(Detector):
    name = "editlens"

    def __init__(self, checkpoint: str = "pangram/editlens_Llama-3.2-3B", threshold: float = 0.5,
                 mirror: str | None = None):
        from huggingface_hub import snapshot_download
        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        adapter_dir = snapshot_download(mirror or checkpoint)
        if mirror:
            verify_mirror(checkpoint, adapter_dir)
        base, self.max_length = BASES.get(checkpoint, (None, 512))
        base = base or PeftConfig.from_pretrained(adapter_dir).base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(base)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        num_labels = self._num_buckets(Path(adapter_dir) / ADAPTER_FILE)
        device = best_device()
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=num_labels, torch_dtype=dtype)
        model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model = PeftModel.from_pretrained(model, adapter_dir).to(device).eval()
        self.threshold = threshold  # recalibrate on the human corpus (run.py calibrate)

    @staticmethod
    def _num_buckets(path: Path) -> int:
        """Read the classifier head size from the adapter weights."""
        from safetensors import safe_open

        with safe_open(str(path), "pt") as f:
            for key in f.keys():
                if key.endswith(("score.weight", "classifier.out_proj.weight", "score.modules_to_save.default.weight",
                                 "classifier.modules_to_save.default.out_proj.weight")):
                    return f.get_slice(key).get_shape()[0]
        raise RuntimeError(f"Impossible de lire le nombre de classes de {path}")

    @torch.inference_mode()
    def detect(self, text: str) -> DetectionResult:
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        probs = torch.softmax(self.model(**enc).logits[0].float(), -1)
        n = probs.numel()
        positions = torch.linspace(0, 1, n, device=probs.device)
        score = float((probs * positions).sum())  # expected bucket, 0 = human, 1 = fully AI
        return DetectionResult(self.name, score, score >= self.threshold,
                               {"score": score, "bucket": int(probs.argmax()), "threshold": self.threshold})
