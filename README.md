# detbench

Banc d'essai de recherche : à quel point les détecteurs de texte IA résistent-ils à
un texte écrit par Claude dans un style humain précis, puis révisé à partir de leurs
propres signaux ? Le banc mesure à la fois la détection **et** le coût en qualité
(fidélité aux données, clarté, naturel), sur le modèle des protocoles de RAID et StealthRL.

## Ce qu'il fait

```
données (brief) ─► Claude (claude -p) ─► texte
                     ▲   style YAML + extraits humains
                     │
         condition style_revise : réécriture des 20 % de phrases
         les plus signalées par les détecteurs locaux, avec garde-fou de sens
                     │
texte ─► détecteurs locaux (Binoculars, Fast-DetectGPT, classifieur HF)
      ─► juge externe non vu par la boucle (Pangram)
      ─► juge qualité (Claude, à l'aveugle) ─► results/runs.jsonl ─► results/report.md
```

Trois conditions sont comparées sur les mêmes briefs :

| Condition | Description |
|---|---|
| `raw` | Prompt simple, sans style |
| `style` | Style YAML + extraits humains en few-shot |
| `style_revise` | `style` + révision ciblée guidée par les détecteurs locaux |

Le corpus humain (`data/human/`) donne le taux de faux positifs de chaque détecteur
et sert à caler les seuils locaux à 1 % de faux positifs.

## Installation (Mac Apple Silicon ou GPU NVIDIA)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # y mettre PANGRAM_API_KEY
claude login                # Claude Code connecté à ton abonnement
```

Le périphérique est choisi automatiquement : CUDA, sinon MPS (Apple Silicon), sinon CPU.
Sur Mac, la paire Qwen2.5-7B / Qwen2.5-7B-Instruct tourne en float16 et occupe environ
30 Go de mémoire unifiée ; sous 32 Go, utiliser la paire Qwen2.5-3B. Sur GPU NVIDIA,
la quantification 8 bits la ramène à environ 16 Go. Les deux détecteurs locaux
partagent cette paire, elle n'est chargée qu'une fois.

## Utilisation

```bash
# 1. Corpus humain → faux positifs, puis seuils calés à 1 % de FPR
python -m detbench.run human
python -m detbench.run calibrate

# 2. Génération + mesures (petit essai d'abord, sans Pangram ni juge)
python -m detbench.run generate --brief velo-partage --style technicien --no-external --no-judge

# 3. Campagne complète
python -m detbench.run generate --repeats 3

# 4. Rapport
python -m detbench.run report
```

## Ajouter un style

1. Créer `styles/<nom>.yaml` (voir `styles/technicien.yaml`).
2. Déposer 5 à 15 extraits humains dans `styles/samples/<nom>/*.txt`.
   Les repères chiffrés (longueur et variance des phrases…) sont calculés dessus.

## Coûts

| Brique | Coût |
|---|---|
| Rédaction et juge qualité | inclus dans l'abonnement Claude (quotas d'usage individuel) |
| Binoculars, Fast-DetectGPT, classifieur | gratuit, GPU local |
| Pangram | gratuit jusqu'à 2 000 mots/jour ; budget journalier appliqué et résultats mis en cache |

## Limites connues

- Les seuils Binoculars publiés ne valent que pour la paire Falcon-7B : toute autre paire
  doit passer par `calibrate`.
- Le classifieur HF par défaut est entraîné sur de l'anglais.
- Pangram 4 est mis à jour régulièrement : noter la `version` renvoyée (stockée dans chaque résultat).
- Turnitin n'est accessible qu'aux établissements ; il n'est pas intégré.

## Références

- Hans et al., *Spotting LLMs With Binoculars*, ICML 2024 — github.com/ahans30/Binoculars
- Bao et al., *Fast-DetectGPT*, ICLR 2024 — github.com/baoguangsheng/fast-detect-gpt
- Dugan et al., *RAID*, ACL 2024 — github.com/liamdugan/raid
- Cheng et al., *Adversarial Paraphrasing*, NeurIPS 2025 — github.com/chengez/Adversarial-Paraphrasing
- *StealthRL*, arXiv 2602.08934 — github.com/suraj-ranganath/StealthRL
