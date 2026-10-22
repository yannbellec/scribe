"""Aggregate runs.jsonl (+ human.jsonl) into a Markdown report."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def _auroc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _fmt(x, pct=False):
    if x is None:
        return "–"
    return f"{100 * x:.0f} %" if pct else f"{x:.2f}"


def build_report(results_dir: Path) -> Path:
    runs = _load(results_dir / "runs.jsonl")
    human = _load(results_dir / "human.jsonl")
    detectors = sorted({d for r in runs + human for d, v in r["detectors"].items() if "p_ai" in v})

    groups = defaultdict(list)
    for r in runs:
        groups[(r.get("language", "fr"), r["condition"], r["style"] or "—")].append(r)
    for r in human:
        groups[(r.get("language", "fr"), "humain", "corpus")].append(r)

    lines = ["# Résultats du banc d'essai", "",
             f"{len(runs)} textes générés, {len(human)} textes humains.", "",
             "## Taux de signalement par détecteur", "",
             "Part des textes que chaque détecteur classe IA à son seuil. Sur la ligne « humain », "
             "c'est le taux de faux positifs.", "",
             "| " + " | ".join(["Langue", "Condition", "Style", "n", *detectors, "Faits", "Clarté", "Naturel"]) + " |",
             "|" + "---|" * (7 + len(detectors))]
    for (lang, cond, style), recs in sorted(groups.items()):
        cells = []
        for d in detectors:
            vals = [r["detectors"][d]["flagged"] for r in recs if "flagged" in r["detectors"].get(d, {})]
            cells.append(_fmt(sum(vals) / len(vals), pct=True) if vals else "–")
        q = [r.get("quality") or {} for r in recs]
        quality = [_fmt(_mean([x.get("fact_recall") for x in q]), pct=True),
                   _fmt(_mean([x.get("clarte") for x in q])), _fmt(_mean([x.get("naturel") for x in q]))]
        lines.append("| " + " | ".join([lang, cond, style, str(len(recs)), *cells, *quality]) + " |")

    if human:
        lines += ["", "## Séparabilité humain / IA (AUROC)", "",
                  "0,5 = le détecteur ne distingue plus les textes de la condition des textes humains.", "",
                  "| Langue | Condition | " + " | ".join(detectors) + " |", "|---|---|" + "---|" * len(detectors)]
        lang_of = lambda r: r.get("language", "fr")
        for lang, cond in sorted({(lang_of(r), r["condition"]) for r in runs}):
            cells = []
            for d in detectors:
                pos = [r["detectors"][d]["p_ai"] for r in runs
                       if lang_of(r) == lang and r["condition"] == cond and "p_ai" in r["detectors"].get(d, {})]
                neg = [r["detectors"][d]["p_ai"] for r in human
                       if lang_of(r) == lang and "p_ai" in r["detectors"].get(d, {})]
                cells.append(_fmt(_auroc(pos, neg)))
            lines.append(f"| {lang} | {cond} | " + " | ".join(cells) + " |")

    revs = [r for r in runs if r.get("revision")]
    if revs:
        n_rounds = _mean([len(r["revision"]["rounds"]) - 1 for r in revs])
        reverted = sum(any(x.get("reverted") for x in r["revision"]["rounds"]) for r in revs)
        lines += ["", "## Boucle de révision", "",
                  f"- Tours de réécriture en moyenne : {_fmt(n_rounds)}",
                  f"- Textes où le garde-fou de sens a annulé au moins un tour : {reverted} / {len(revs)}"]

    path = results_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
