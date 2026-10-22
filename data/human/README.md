Corpus humain de référence : 30 à 100 textes `.txt` écrits par des humains,
du même genre et de la même longueur que les textes générés, idéalement
publiés avant 2022 pour être sûr qu'ils ne viennent pas d'une IA.
Il sert à mesurer le taux de faux positifs de chaque détecteur et à caler
les seuils locaux (`python -m detbench.run calibrate`).
