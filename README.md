# Physique statistique et machine learning

> **EN —** Projects at the interface of statistical physics, data analysis and machine learning, carried out at ESPCI Paris - PSL. First project: unsupervised identification of bird species from their song, with a hand-written short-time Fourier transform and a principal component analysis coded from scratch (SVD, no ML library). Python · NumPy · SciPy · Matplotlib.

Projets réalisés à l'**ESPCI Paris - PSL**, à la frontière entre physique statistique, analyse de données et apprentissage automatique.

| Projet | Sujet | Méthodes | Résultat clé |
|---|---|---|---|
| [01 · Le chant des oiseaux](01-chant-des-oiseaux-acp) | Reconnaître une espèce d'oiseau à partir d'un enregistrement | STFT codée avec NumPy, ACP par SVD sans bibliothèque de ML, plus proche voisin | 30 enregistrements répartis en 3 groupes bien séparés dans le plan (PC1, PC2), qui portent 62,5 % de l'inertie |

<p align="center">
  <img src="01-chant-des-oiseaux-acp/figures/classification_finale.png" width="60%" alt="Classification des 30 enregistrements dans le plan des deux premières composantes principales">
</p>

## Lancer les scripts

```bash
pip install -r requirements.txt
python 01-chant-des-oiseaux-acp/visualisation_stft.py chemin/vers/les/enregistrements
python 01-chant-des-oiseaux-acp/identification_acp.py chemin/vers/les/enregistrements
```

Les enregistrements audio ne sont pas fournis dans ce dépôt (voir le README du projet).

## Auteur

**Maxime Rimbaud** — élève ingénieur à l'ESPCI Paris - PSL · [LinkedIn](https://www.linkedin.com/in/maxime-rimbaud)

Le projet 01 a été réalisé en binôme avec Romain Bertin. Les énoncés appartiennent à l'ESPCI et ne sont pas reproduits ici.
