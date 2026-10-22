"""Detector registry: builds detectors from the `detectors` section of config.yaml."""
from __future__ import annotations

from .base import DetectionResult, Detector, SentenceScore


def build_detectors(cfg: dict) -> list[Detector]:
    out: list[Detector] = []
    pair = cfg.get("lm_pair", {})
    pair_args = {k: pair[k] for k in ("observer", "performer", "quantization", "max_tokens") if k in pair}
    for name, opts in (cfg.get("detectors") or {}).items():
        opts = dict(opts or {})
        if not opts.pop("enabled", True):
            continue
        if name == "binoculars":
            from .binoculars import BinocularsDetector
            out.append(BinocularsDetector(**pair_args, **opts))
        elif name == "fast_detectgpt":
            from .fast_detectgpt import FastDetectGPTDetector
            out.append(FastDetectGPTDetector(**pair_args, **opts))
        elif name == "hf_classifier":
            from .hf_classifier import HFClassifierDetector
            out.append(HFClassifierDetector(**opts))
        elif name == "editlens":
            from .editlens import EditLensDetector
            out.append(EditLensDetector(**opts))
        elif name == "pangram":
            from .pangram import PangramDetector
            out.append(PangramDetector(**opts))
        else:
            raise ValueError(f"Détecteur inconnu : {name}")
    return out


__all__ = ["DetectionResult", "Detector", "SentenceScore", "build_detectors"]
