"""Detector-feedback revision: rewrite only the sentences local detectors flag most.

Only local detectors drive the loop; external APIs (Pangram) are measured once
at the end so they act as held-out judges, as in StealthRL's transfer setting.
"""
from __future__ import annotations

import statistics as stats
from dataclasses import dataclass, field

from .detectors import DetectionResult, Detector
from .llm import Claude
from .styles import Style
from .text_utils import split_sentences


@dataclass
class RevisionTrace:
    rounds: list[dict] = field(default_factory=list)


def _zscores(values: list[float]) -> list[float]:
    if len(values) < 2:
        return [0.0] * len(values)
    mu, sd = stats.mean(values), stats.pstdev(values) or 1.0
    return [(v - mu) / sd for v in values]


def rank_sentences(text: str, results: list[DetectionResult]) -> list[tuple[float, str, int, int]]:
    """Combine per-sentence scores of all detectors (z-scored per detector)."""
    spans = split_sentences(text)
    combined = {(a, b): [] for _, a, b in spans}
    for r in results:
        if not r.sentences:
            continue
        zs = _zscores([s.ai_score for s in r.sentences])
        for s, z in zip(r.sentences, zs):
            if (s.start, s.end) in combined:
                combined[(s.start, s.end)].append(z)
    ranked = [(stats.mean(combined[(a, b)]) if combined[(a, b)] else 0.0, sent, a, b) for sent, a, b in spans]
    return sorted(ranked, key=lambda x: -x[0])


def _rewrite(llm: Claude, style: Style, text: str, targets: list[tuple[float, str, int, int]],
             language: str = "fr") -> str:
    listing = "\n".join(f'{i}. "{t[1]}"' for i, t in enumerate(targets, 1))
    prompt = (
        f"Voici un texte :\n<texte>\n{text}\n</texte>\n\n"
        f"Réécris uniquement les phrases suivantes pour qu'elles sonnent davantage comme le style demandé : "
        f"rythme moins régulier, mots plus concrets, formulations moins attendues. Garde exactement les mêmes "
        f"informations et la cohérence avec les phrases voisines. Tu peux fusionner ou couper une phrase.\n"
        f"{listing}\n\n"
        + ("Les nouvelles phrases restent en anglais, comme le texte.\n" if language == "en" else "")
        + 'Réponds en JSON : {"1": "nouvelle phrase", "2": "..."}')
    out = llm.complete_json(prompt, style.system_prompt(language))
    new_text = text
    # Replace from the end so earlier offsets stay valid.
    for i, (_, sent, a, b) in sorted(enumerate(targets, 1), key=lambda x: -x[1][2]):
        replacement = str(out.get(str(i), "")).strip()
        if replacement and new_text[a:b] == sent:
            new_text = new_text[:a] + replacement + new_text[b:]
    return new_text


def revise(text: str, llm: Claude, style: Style, detectors: list[Detector], max_rounds: int = 3,
           fraction: float = 0.2, min_similarity: float = 0.88, similarity_fn=None,
           language: str = "fr") -> tuple[str, RevisionTrace]:
    trace = RevisionTrace()
    original = text
    # Every local detector decides when to stop; only those with sentence scores
    # (Binoculars, Fast-DetectGPT) choose which sentences to rewrite.
    for rnd in range(max_rounds + 1):
        results = [d.detect(text) for d in detectors]
        entry = {"round": rnd, "scores": {r.detector: round(r.ai_probability, 4) for r in results},
                 "flagged": [r.detector for r in results if r.flagged]}
        trace.rounds.append(entry)
        if not entry["flagged"] or rnd == max_rounds:
            break
        ranked = rank_sentences(text, results)
        k = max(1, round(len(ranked) * fraction))
        candidate = _rewrite(llm, style, text, ranked[:k], language)
        if similarity_fn is not None:
            sim = similarity_fn(original, candidate)
            entry["similarity_after"] = round(sim, 4)
            if sim < min_similarity:
                entry["reverted"] = True  # quality gate: meaning drifted too far
                continue
        text = candidate
    return text, trace
