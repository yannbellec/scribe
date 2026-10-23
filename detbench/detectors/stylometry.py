"""Model-free style metrics. Not a detector verdict: used to compare against style targets."""
from __future__ import annotations

import re
import statistics as stats

from ..text_utils import split_sentences

# Stock phrases over-represented in LLM output (FR + EN). Extend per style.
DEFAULT_AI_PHRASES = [
    "il est important de noter", "il convient de souligner", "en conclusion", "en somme",
    "dans un monde en constante évolution", "jouer un rôle crucial", "un véritable", "force est de constater",
    "par ailleurs", "de plus", "en outre", "il est essentiel", "n'est pas seulement", "mais aussi",
    "plonger dans", "à l'ère du numérique", "un équilibre délicat", "tapisserie",
    "it is important to note", "in today's", "delve", "tapestry", "landscape", "furthermore",
    "moreover", "in conclusion", "not only", "plays a crucial role", "navigate the complexities",
]


def style_metrics(text: str, ai_phrases: list[str] | None = None) -> dict:
    sents = [s for s, _, _ in split_sentences(text)]
    lengths = [len(re.findall(r"\w+", s)) for s in sents] or [0]
    words = re.findall(r"\w+", text.lower())
    lower = text.lower()
    phrases = ai_phrases or DEFAULT_AI_PHRASES
    hits = {p: lower.count(p) for p in phrases if p in lower}
    mean = stats.mean(lengths)
    return {
        "n_words": len(words),
        "n_sentences": len(sents),
        "sent_len_mean": round(mean, 2),
        "sent_len_std": round(stats.pstdev(lengths), 2),
        # burstiness: coefficient of variation of sentence length (human prose tends to be higher)
        "burstiness": round(stats.pstdev(lengths) / mean, 3) if mean else 0.0,
        "type_token_ratio": round(len(set(words)) / len(words), 3) if words else 0.0,
        "ai_phrase_per_100w": round(100 * sum(hits.values()) / max(len(words), 1), 3),
        "ai_phrase_hits": hits,
        "em_dash_per_100w": round(100 * text.count("—") / max(len(words), 1), 3),
        "questions_ratio": round(sum(s.endswith("?") for s in sents) / max(len(sents), 1), 3),
    }
