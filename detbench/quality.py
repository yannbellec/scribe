"""Quality measures: fact fidelity and readability judged by Claude, semantic drift by embeddings."""
from __future__ import annotations

from functools import lru_cache

from .briefs import Brief
from .llm import Claude

JUDGE_SYSTEM = (
    "Tu es un relecteur exigeant. Tu évalues un texte par rapport aux données qu'il devait contenir. "
    "Tu ne sais pas comment le texte a été produit et tu ne cherches pas à le deviner. "
    "Réponds uniquement en JSON.")


def judge(llm: Claude, brief: Brief, text: str) -> dict:
    facts = "\n".join(f"{i}. {f}" for i, f in enumerate(brief.facts, 1))
    prompt = (
        f"Données attendues :\n{facts}\n\nTexte :\n<texte>\n{text}\n</texte>\n\n"
        "Pour chaque donnée, indique present, absent ou contredit. Puis note de 1 à 5 :\n"
        "- clarte : le texte se comprend du premier coup ;\n"
        "- naturel : la langue est fluide et vivante pour un lecteur francophone ;\n"
        "- coherence : les idées s'enchaînent logiquement.\n"
        'Format : {"facts": {"1": "present", ...}, "clarte": n, "naturel": n, "coherence": n, '
        '"remarque": "une phrase"}')
    out = llm.complete_json(prompt, JUDGE_SYSTEM)
    statuses = list((out.get("facts") or {}).values())
    out["fact_recall"] = round(sum(s == "present" for s in statuses) / max(len(brief.facts), 1), 3)
    out["fact_contradictions"] = sum(s == "contredit" for s in statuses)
    return out


@lru_cache(maxsize=1)
def _embedder(model: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model)


def semantic_similarity(a: str, b: str, model: str = "intfloat/multilingual-e5-base") -> float:
    emb = _embedder(model).encode([f"passage: {a}", f"passage: {b}"], normalize_embeddings=True)
    return float(emb[0] @ emb[1])
