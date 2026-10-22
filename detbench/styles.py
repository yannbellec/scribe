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
    samples: list[str] = field(default_factory=list)
    targets: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "Style":
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        samples = []
        sample_dir = path.parent / "samples" / path.stem
        if sample_dir.is_dir():
            samples = [p.read_text(encoding="utf-8").strip() for p in sorted(sample_dir.glob("*.txt"))]
        style = cls(name=data.get("name", path.stem), description=data.get("description", ""),
                    voice=data.get("voice", ""), rules=data.get("rules", []),
                    banned_phrases=data.get("banned_phrases", []), samples=samples,
                    targets=data.get("targets") or {})
        if samples and not style.targets:
            style.targets = style.measured_targets()
        return style

    def measured_targets(self) -> dict:
        """Average style metrics over the human samples."""
        ms = [style_metrics(s) for s in self.samples]
        keys = ("sent_len_mean", "sent_len_std", "burstiness", "type_token_ratio", "em_dash_per_100w",
                "questions_ratio")
        return {k: round(sum(m[k] for m in ms) / len(ms), 3) for k in keys}

    def system_prompt(self, max_samples: int = 4) -> str:
        parts = [f"Tu écris dans le style « {self.name} ». {self.description}".strip(),
                 f"Voix : {self.voice}".strip()]
        if self.rules:
            parts.append("Règles d'écriture :\n" + "\n".join(f"- {r}" for r in self.rules))
        if self.banned_phrases:
            parts.append("Tournures à ne jamais employer : " + ", ".join(f"« {p} »" for p in self.banned_phrases))
        if self.targets:
            t = self.targets
            parts.append(
                "Repères chiffrés mesurés sur de vrais textes de ce style : "
                f"phrases de {t.get('sent_len_mean', '?')} mots en moyenne avec un écart-type de "
                f"{t.get('sent_len_std', '?')} mots (alterne phrases très courtes et longues).")
        if self.samples:
            shots = "\n\n".join(f"<extrait>\n{s}\n</extrait>" for s in self.samples[:max_samples])
            parts.append("Extraits authentiques à imiter pour le rythme, le lexique et le ton "
                         "(ne pas copier leur contenu) :\n" + shots)
        parts.append("Réponds uniquement avec le texte demandé, sans titre, sans préambule ni commentaire.")
        return "\n\n".join(p for p in parts if p)


def load_styles(directory: str | Path) -> dict[str, Style]:
    return {p.stem: Style.load(p) for p in sorted(Path(directory).glob("*.yaml"))}
