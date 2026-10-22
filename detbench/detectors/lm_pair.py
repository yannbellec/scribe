"""Shared base/instruct model pair used by Binoculars and Fast-DetectGPT.

Both detectors need the logits of two causal LMs sharing one tokenizer on the
same text, so one pair is loaded once and its per-token statistics are cached.

References
- Binoculars (Hans et al., ICML 2024), github.com/ahans30/Binoculars
- Fast-DetectGPT (Bao et al., ICLR 2024), github.com/baoguangsheng/fast-detect-gpt
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F

from ..device import best_device


@dataclass
class TokenStats:
    """Per-token statistics, one entry per predicted token."""
    offsets: list[tuple[int, int]]  # character span of each predicted token
    ce: torch.Tensor        # -log p_performer(x_t)              (Binoculars log-PPL)
    xent: torch.Tensor      # H(p_observer, p_performer)          (Binoculars cross-PPL)
    ll: torch.Tensor        # log p_scoring(x_t), scoring = performer (Fast-DetectGPT)
    mean_ref: torch.Tensor  # E_{x~p_ref}[log p_scoring(x)]
    var_ref: torch.Tensor   # Var_{x~p_ref}[log p_scoring(x)]


class LMPair:
    def __init__(self, observer: str, performer: str, quantization: str | None = "8bit",
                 max_tokens: int = 512, device_map: str = "auto"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(observer)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        device = best_device()
        if quantization in ("8bit", "4bit") and device != "cuda":
            # bitsandbytes needs CUDA; on Apple Silicon / CPU load unquantized instead.
            warnings.warn(f"Quantification {quantization} ignorée sur {device} (CUDA requis)")
            quantization = None
        if device == "cuda":
            kwargs: dict = {"device_map": device_map}
            if quantization in ("8bit", "4bit"):
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=quantization == "8bit", load_in_4bit=quantization == "4bit")
            else:
                kwargs["torch_dtype"] = torch.bfloat16
            load = lambda name: AutoModelForCausalLM.from_pretrained(name, **kwargs).eval()
        else:
            dtype = torch.float16 if device == "mps" else torch.float32
            load = lambda name: AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device).eval()
        self.observer = load(observer)
        self.performer = load(performer)
        self.max_tokens = max_tokens
        self._cache: dict[str, TokenStats] = {}

    @torch.inference_mode()
    def _window(self, text: str, start: int, end: int):
        piece = text[start:end]
        enc = self.tokenizer(piece, return_tensors="pt", return_offsets_mapping=True,
                             truncation=True, max_length=self.max_tokens)
        ids = enc["input_ids"].to(self.observer.device)
        offsets = [(a + start, b + start) for a, b in enc["offset_mapping"][0].tolist()]
        obs = self.observer(ids).logits[0, :-1].float()
        perf = self.performer(ids.to(self.performer.device)).logits[0, :-1].float().to(obs.device)
        labels = ids[0, 1:]

        # Binoculars: performer perplexity vs observer→performer cross-entropy
        perf_lp = F.log_softmax(perf, -1)
        ce = -perf_lp.gather(-1, labels[:, None]).squeeze(-1)
        xent = -(F.softmax(obs, -1) * perf_lp).sum(-1)

        # Fast-DetectGPT (analytic): reference = observer (base), scoring = performer
        # (instruct), the paper's best setting (falcon-7b → falcon-7b-instruct)
        score_lp = perf_lp
        ref_p = F.softmax(obs, -1)
        ll = score_lp.gather(-1, labels[:, None]).squeeze(-1)
        mean_ref = (ref_p * score_lp).sum(-1)
        var_ref = (ref_p * score_lp.pow(2)).sum(-1) - mean_ref.pow(2)
        return offsets[1:], ce.cpu(), xent.cpu(), ll.cpu(), mean_ref.cpu(), var_ref.cpu()

    def stats(self, text: str) -> TokenStats:
        if text in self._cache:
            return self._cache[text]
        # Window over the text by characters so long documents stay within max_tokens.
        approx_chars = self.max_tokens * 3
        parts = []
        pos = 0
        while pos < len(text):
            end = min(len(text), pos + approx_chars)
            if end < len(text):
                cut = text.rfind(" ", pos, end)
                end = cut if cut > pos else end
            parts.append(self._window(text, pos, end))
            pos = end
        offsets = [o for p in parts for o in p[0]]
        cat = [torch.cat([p[i] for p in parts]) for i in range(1, 6)]
        result = TokenStats(offsets, *cat)
        if len(self._cache) > 256:
            self._cache.clear()
        self._cache[text] = result
        return result


@lru_cache(maxsize=2)
def get_pair(observer: str, performer: str, quantization: str | None, max_tokens: int) -> LMPair:
    return LMPair(observer, performer, quantization, max_tokens)


def token_mask(offsets: list[tuple[int, int]], start: int, end: int) -> torch.Tensor:
    """Boolean mask of tokens whose span starts inside [start, end)."""
    return torch.tensor([start <= a < end for a, _ in offsets], dtype=torch.bool)
