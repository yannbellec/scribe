"""Input data ("briefs"): the facts a text must convey."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Brief:
    id: str
    subject: str
    facts: list[str]
    audience: str = "grand public"
    genre: str = "article"
    length_words: int = 400
    language: str = "fr"
    extra: dict = field(default_factory=dict)

    def prompt(self) -> str:
        facts = "\n".join(f"- {f}" for f in self.facts)
        if self.language == "en":
            return (f"Write a {self.genre} of about {self.length_words} words in English "
                    f"about: {self.subject}.\nAudience: {self.audience}.\n"
                    f"Include all of these facts faithfully, and invent no others:\n{facts}")
        return (f"Écris un {self.genre} d'environ {self.length_words} mots en français "
                f"sur : {self.subject}.\nPublic : {self.audience}.\n"
                f"Données à intégrer fidèlement (toutes, sans en inventer d'autres) :\n{facts}")


def load_briefs(directory: str | Path) -> list[Brief]:
    briefs = []
    for p in sorted(Path(directory).glob("*.yaml")):
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        known = {k: data.pop(k) for k in list(data) if k in Brief.__dataclass_fields__}
        known.setdefault("id", p.stem)
        briefs.append(Brief(**known, extra=data))
    return briefs
