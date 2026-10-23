"""Sentence splitting shared by detectors, stylometry and the revision loop."""
from __future__ import annotations

import re

# Split after . ! ? … followed by whitespace and an uppercase/quote/digit start.
_SPLIT = re.compile(r"(?<=[.!?…])[\"»”)]?\s+(?=[«\"“(A-ZÀ-ÖØ-Þ0-9])")


def split_sentences(text: str) -> list[tuple[str, int, int]]:
    """Return (sentence, start, end) spans covering the non-blank text."""
    spans = []
    for para_match in re.finditer(r"[^\n]+", text):
        para, offset = para_match.group(), para_match.start()
        pos = 0
        for m in _SPLIT.finditer(para):
            chunk = para[pos:m.start()].strip()
            if chunk:
                s = offset + para.index(chunk, pos)
                spans.append((chunk, s, s + len(chunk)))
            pos = m.end()
        chunk = para[pos:].strip()
        if chunk:
            s = offset + para.index(chunk, pos)
            spans.append((chunk, s, s + len(chunk)))
    return spans


def word_count(text: str) -> int:
    return len(re.findall(r"\w+", text))
