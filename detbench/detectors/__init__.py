"""Detector registry: builds detectors from the `detectors` section of config.yaml."""
from __future__ import annotations

import warnings

from .base import DetectionResult, Detector, SentenceScore


def build_detectors(cfg: dict) -> list[Detector]:
    out: list[Detector] = []
    pair = cfg.get("lm_pair", {})
    pair_args = {k: pair[k] for k in ("observer", "performer", "quantization", "max_tokens") if k in pair}
    for name, opts in (cfg.get("detectors") or {}).items():
        opts = dict(opts or {})
        if not opts.pop("enabled", True):
            continue
        try:
            out.append(_build(name, opts, pair_args))
        except Exception as e:  # e.g. gated model not yet approved: run with the others
            warnings.warn(f"Détecteur {name} ignoré : {e}")
    if not out:
        raise RuntimeError("Aucun détecteur n'a pu être chargé")
    return out


def _build(name: str, opts: dict, pair_args: dict) -> Detector:
    if name == "binoculars":
        from .binoculars import BinocularsDetector
        return BinocularsDetector(**pair_args, **opts)
    if name == "fast_detectgpt":
        from .fast_detectgpt import FastDetectGPTDetector
        return FastDetectGPTDetector(**pair_args, **opts)
    if name == "hf_classifier":
        from .hf_classifier import HFClassifierDetector
        return HFClassifierDetector(**opts)
    if name == "editlens":
        from .editlens import EditLensDetector
        return EditLensDetector(**opts)
    if name == "pangram":
        from .pangram import PangramDetector
        return PangramDetector(**opts)
    raise ValueError(f"Détecteur inconnu : {name}")


__all__ = ["DetectionResult", "Detector", "SentenceScore", "build_detectors"]
