Corpus humain de référence, un sous-dossier par langue : `data/human/fr/`, `data/human/en/`.
Il sert à mesurer le taux de faux positifs de chaque détecteur et à caler les seuils
locaux séparément pour chaque langue (`python -m detbench.run calibrate`).

Le remplir automatiquement (MLSUM pour le français, RAID pour l'anglais) :

    python scripts/fetch_human_corpus.py

On peut aussi y ajouter ses propres textes `.txt`, écrits par des humains et de
préférence publiés avant 2022.
