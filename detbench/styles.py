"""Writing styles: YAML profile + real human samples, turned into a system prompt."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .detectors.stylometry import style_metrics


@dataclass
class Style:
    name: str
    description: str
    voice: str
    rules: list[str] = field(default_factory=list)
    banned_phrases: list[str] = field(default_factory=list)
    samples: dict[str, list[str]] = field(default_factory=dict)  # language → human excerpts
    targets: dict[str, dict] = field(default_factory=dict)       # language → style metrics

    @classmethod
    def load(cls, path: str | Path) -> "Style":
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        # Excerpts live in samples/<style>/<lang>/*.txt (fr, en).
        samples = {}
        sample_dir = path.parent / "samples" / path.stem
        for lang_dir in sorted(p for p in sample_dir.glob("*") if p.is_dir()):
            texts = [p.read_text(encoding="utf-8").strip() for p in sorted(lang_dir.glob("*.txt"))]
            if texts:
                samples[lang_dir.name] = texts
        style = cls(name=data.get("name", path.stem), description=data.get("description", ""),
                    voice=data.get("voice", ""), rules=data.get("rules", []),
                    banned_phrases=data.get("banned_phrases", []), samples=samples,
                    targets=data.get("targets") or {})
        for lang, texts in samples.items():
            style.targets.setdefault(lang, style.measured_targets(texts))
        return style

    @staticmethod
    def measured_targets(texts: list[str]) -> dict:
        """Average style metrics over human excerpts."""
        ms = [style_metrics(s) for s in texts]
        keys = ("sent_len_mean", "sent_len_std", "burstiness", "type_token_ratio", "em_dash_per_100w",
                "questions_ratio")
        return {k: round(sum(m[k] for m in ms) / len(ms), 3) for k in keys}

    def system_prompt(self, language: str = "fr", max_samples: int = 4) -> str:
        parts = [f"Tu écris dans le style « {self.name} ». {self.description}".strip(),
                 f"Voix : {self.voice}".strip()]
        if self.rules:
            parts.append("Règles d'écriture :\n" + "\n".join(f"- {r}" for r in self.rules))
        if self.banned_phrases:
            parts.append("Tournures à ne jamais employer : " + ", ".join(f"« {p} »" for p in self.banned_phrases))
        if language == "en":
            parts.append("Le texte final est en anglais : applique ces consignes à l'anglais, "
                         "avec ses propres tournures toutes faites à éviter (delve, tapestry, "
                         "it's worth noting, in today's fast-paced world, moreover, furthermore).")
        t = self.targets.get(language)
        if t:
            parts.append(
                "Repères chiffrés mesurés sur de vrais textes de ce style : "
                f"phrases de {t.get('sent_len_mean', '?')} mots en moyenne avec un écart-type de "
                f"{t.get('sent_len_std', '?')} mots (alterne phrases très courtes et longues).")
        if self.samples.get(language):
            shots = "\n\n".join(f"<extrait>\n{s}\n</extrait>" for s in self.samples[language][:max_samples])
            parts.append("Extraits authentiques à imiter pour le rythme, le lexique et le ton "
                         "(ne pas copier leur contenu) :\n" + shots)
        parts.append("Réponds uniquement avec le texte demandé, sans titre, sans préambule ni commentaire.")
        return "\n\n".join(p for p in parts if p)


def load_styles(directory: str | Path) -> dict[str, Style]:
    return {p.stem: Style.load(p) for p in sorted(Path(directory).glob("*.yaml"))}
