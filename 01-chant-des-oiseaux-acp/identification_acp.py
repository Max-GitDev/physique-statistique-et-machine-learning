#!/usr/bin/env python3
"""
Le chant des oiseaux - Partie 2 : identification automatique par analyse en composantes principales.

Chaque enregistrement est résumé par son spectre moyen normalisé (STFT, bande 200-6000 Hz).
L'ACP est codée à la main par décomposition en valeurs singulières (sans scikit-learn).
Trois enregistrements étiquetés servent ensuite à attribuer une espèce aux 27 autres,
par plus proche voisin dans le plan (PC1, PC2).

Auteurs : Maxime Rimbaud et Romain Bertin - ESPCI Paris - PSL (mars 2026).
Usage : python identification_acp.py [dossier_des_enregistrements]
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as scipy_signal
from scipy.io import wavfile

ICI = Path(__file__).parent
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else ICI / "data"   # dossier contenant part1/ et part2/
FIG = ICI / "figures"
FIG.mkdir(exist_ok=True)


# QUESTION 1 — Chargement des 30 fichiers .wav


signals = []   # liste des signaux (float)
sample_rates = []

for i in range(30):
    fs, data = wavfile.read(DATA / "part2" / f"sample_{i}.wav")
    # Mono : on prend le canal gauche si stéréo
    if data.ndim > 1:
        data = data[:, 0]
    signals.append(data.astype(float))
    sample_rates.append(fs)
    print(f"sample_{i:02d}: fs={fs} Hz, durée={len(data)/fs:.2f}s")


# QUESTION 2 — Construction de la matrice de descripteurs A et PCA


# ── 2.1 Calcul de la signature spectrale de chaque échantillon ───────────────

FREQ_MIN = 200    # Hz — borne basse de la bande utile
FREQ_MAX = 6000   # Hz — borne haute de la bande utile
N_S      = 1024   # taille de segment pour la STFT (2^10)

def compute_descriptor(signal, fs, freq_min=FREQ_MIN, freq_max=FREQ_MAX, nperseg=N_S):
    """
    Calcule le vecteur de descripteurs x_i d'un enregistrement.
    """
    # STFT scipy avec fenêtre de Hann (autorisée en partie 2)
    freqs, times, Z = scipy_signal.stft(signal, fs=fs,
                                        nperseg=nperseg,
                                        noverlap=nperseg // 2)

    # |A(f, t)| : amplitude du spectrogramme
    A = np.abs(Z)  # shape (n_freqs, n_times)

    # Moyenne temporelle : spectre moyen se_i
    se = A.mean(axis=1)  # shape (n_freqs,)

    # Sélection de la bande de fréquences utile
    mask = (freqs >= freq_min) & (freqs <= freq_max)
    se_sel    = se[mask]
    freqs_sel = freqs[mask]

    # Normalisation par la std (équation 2)
    x_i = se_sel / np.std(se_sel)

    return x_i, freqs_sel

# Calcul des descripteurs pour les 30 échantillons
descriptors = []
freqs_ref   = None   # grille de fréquences commune (identique pour tous)

for i, (sig, fs) in enumerate(zip(signals, sample_rates)):
    x_i, freqs_sel = compute_descriptor(sig, fs)
    descriptors.append(x_i)
    if freqs_ref is None:
        freqs_ref = freqs_sel

# ── Harmonisation de la dimension des descripteurs ───────────────────────────

n_ref = len(freqs_ref)
A_mat = np.zeros((30, n_ref))

for i, (x_i, fs) in enumerate(zip(descriptors, sample_rates)):
    _, freqs_i = compute_descriptor(signals[i], fs)
    # Réinterpolation sur la grille de référence
    A_mat[i, :] = np.interp(freqs_ref, freqs_i, x_i,
                          left=x_i[0], right=x_i[-1])

print(f"\nMatrice de descripteurs A : shape = {A_mat.shape}")
print(f"Chaque ligne = un échantillon, chaque colonne = une fréquence dans [{FREQ_MIN},{FREQ_MAX}] Hz")

# ── 2.2 PCA sans sklearn ──────────────────────────────────────────────────────

def pca(X, center=True, scale=False):
    """
    Analyse en Composantes Principales via décomposition en valeurs singulières.
    """
    X_c = X.copy()

    # Centrage : soustraction de la moyenne de chaque variable
    if center:
        X_c = X_c - X_c.mean(axis=0)

    # Réduction : division par la std de chaque variable (PCA normée)
    if scale:
        std = X_c.std(axis=0)
        std[std == 0] = 1          # éviter division par zéro
        X_c = X_c / std

    # SVD de la matrice centrée : X_c = U · Σ · Vᵀ
    # Les colonnes de Vᵀ sont les vecteurs propres de la matrice de covariance
    U, s, Vt = np.linalg.svd(X_c, full_matrices=False)

    # Valeurs propres ∝ s²
    eigenvalues = s**2

    # Taux d'inertie expliquée par chaque composante
    explained = 100 * eigenvalues / eigenvalues.sum()

    # Coordonnées des individus dans l'espace des CP
    scores = np.einsum("ij,kj->ik", X_c, Vt)      # X_c · V, shape (n_samples, n_components)

    return scores, Vt, explained, X_c



scores, components, explained, A_centered = pca(A_mat, center=True, scale=False)

print("\nTaux d'inertie expliquée par composante :")
for k, e in enumerate(explained[:10]):
    cumul = explained[:k+1].sum()
    print(f"  PC{k+1} : {e:5.1f}%   (cumulé : {cumul:5.1f}%)")


# QUESTION 3 — Affichage de la PCA et identification des espèces


# Labels connus (index 0, 1, 2)
LABELS_CONNUS = {0: "Merle noir", 1: "Rouge-gorge", 2: "Corneille"}
COLORS = {"Merle noir": "#e74c3c", "Rouge-gorge": "#e67e22", "Corneille": "#2c3e50"}
MARKERS = {"Merle noir": "o", "Rouge-gorge": "s", "Corneille": "^"}

# ── Figure 1 : Éboulis des valeurs propres ────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(1, 11), explained[:10], color="steelblue", alpha=0.8, label="Inertie individuelle")
ax.plot(range(1, 11), np.cumsum(explained[:10]), "o-", color="firebrick", label="Inertie cumulée")
ax.axhline(90, color="gray", linestyle="--", alpha=0.5, label="Seuil 90%")
ax.set_xlabel("Composante principale")
ax.set_ylabel("Inertie expliquée (%)")
ax.set_title("Éboulis des valeurs propres ")
ax.legend()
ax.set_xticks(range(1, 11))

plt.tight_layout()
plt.savefig(FIG / "eboulis_valeurs_propres.png", dpi=150)
plt.close()

# ── Figure 2 : Projection PC1 vs PC2 (individus) ─────────────────────────────
# On projette les 30 échantillons dans le plan (PC1, PC2).
# Les points 0, 1, 2 ont des labels connus — on les utilise pour identifier
# à quel groupe appartiennent les autres points.

fig, ax = plt.subplots(figsize=(9, 7))

# Tracé de TOUS les points 
ax.scatter(scores[:, 0], scores[:, 1],
           c="lightgray", s=80, zorder=2, label="Échantillons inconnus")

# Mise en évidence des 3 points connus pour identifier les clusters
for idx, label in LABELS_CONNUS.items():
    ax.scatter(scores[idx, 0], scores[idx, 1],
               c=COLORS[label], marker=MARKERS[label],
               s=200, zorder=5, label=f"échantillon_{idx} ({label})",
               edgecolors="black", linewidths=1.5)
    ax.annotate(f" {idx}", (scores[idx, 0], scores[idx, 1]),
                fontsize=9, color=COLORS[label])


for i in range(30):
    ax.annotate(str(i), (scores[i, 0] + 0.02, scores[i, 1] + 0.02),
                fontsize=7, color="dimgray")

ax.set_xlabel(f"PC1 ({explained[0]:.1f}% d'inertie)")
ax.set_ylabel(f"PC2 ({explained[1]:.1f}% d'inertie)")
ax.set_title("Projection PCA — Plan PC1 × PC2\n(les points connus permettent d'identifier les groupes)")
ax.legend(loc="best", fontsize=9)
ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(0, color="gray", linewidth=0.5)

plt.tight_layout()
plt.savefig(FIG / "projection_pc1_pc2.png", dpi=150)
plt.close()

# ── Attribution automatique des groupes ──────────────────────────────────────
# On assigne chaque échantillon au groupe du point connu le plus proche dans l'espace (PC1, PC2).

known_indices  = list(LABELS_CONNUS.keys())     # [0, 1, 2]
known_labels   = list(LABELS_CONNUS.values())
known_scores   = scores[known_indices, :2]       # coordonnées des 3 points connus

assignments = {}
for i in range(30):
    if i in LABELS_CONNUS:
        assignments[i] = LABELS_CONNUS[i]
    else:
        # Distance euclidienne aux 3 centres connus dans PC1-PC2
        dists = np.linalg.norm(scores[i, :2] - known_scores, axis=1)
        nearest = known_labels[np.argmin(dists)]
        assignments[i] = nearest

# Affichage du tableau d'identification
print("\n── Tableau d'identification ───────────────────────────────")
print(f"{'échantillon':>8} | {'Espèce attribuée'}")
print("-" * 35)
for i in range(30):
    print(f"échantillon_{i:02d} | {assignments[i]}")

# Résumé par espèce
print("\n── Résumé par espèce ──────────────────────────────────────")
for label in ["Merle noir", "Rouge-gorge", "Corneille"]:
    idxs = [i for i, l in assignments.items() if l == label]
    print(f"{label:20s}: échantillon {sorted(idxs)}")

# ── Figure 3 : PCA colorée selon l'espèce identifiée ─────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))

for label in ["Merle noir", "Rouge-gorge", "Corneille"]:
    idxs = [i for i, l in assignments.items() if l == label]
    ax.scatter(scores[idxs, 0], scores[idxs, 1],
               c=COLORS[label], marker=MARKERS[label],
               s=120, label=label, edgecolors="black",
               linewidths=0.8, zorder=3)
    for i in idxs:
        ax.annotate(str(i), (scores[i, 0] + 0.02, scores[i, 1] + 0.02),
                    fontsize=7, color=COLORS[label])

ax.set_xlabel(f"PC1 ({explained[0]:.1f}% d'inertie)")
ax.set_ylabel(f"PC2 ({explained[1]:.1f}% d'inertie)")
ax.set_title("PCA — Classification finale des 30 échantillons")
ax.legend(fontsize=10)
ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(0, color="gray", linewidth=0.5)
plt.tight_layout()
plt.savefig(FIG / "classification_finale.png", dpi=150)
plt.close()

# ── Figure 4 : Vecteurs propres PC1 et PC2 (loadings) ────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

for k, ax in enumerate(axes):
    ax.plot(freqs_ref, components[k], color=f"C{k}", linewidth=1.5)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_ylabel(f"Chargement PC{k+1}")
    ax.set_title(f"PC{k+1} — vecteur propre ({explained[k]:.1f}% d'inertie)")
    ax.set_xlim(FREQ_MIN, FREQ_MAX)

axes[-1].set_xlabel("Fréquence (Hz)")
plt.suptitle("Vecteurs propres des deux premières composantes", fontsize=12)
plt.tight_layout()
plt.savefig(FIG / "vecteurs_propres.png", dpi=150)
plt.close()

# ── Figure 5 : Spectres moyens des 3 espèces (pour interpréter la PCA) ───────
fig, ax = plt.subplots(figsize=(10, 5))

for label, color in COLORS.items():
    idxs = [i for i, l in assignments.items() if l == label]
    mean_spec = A_mat[idxs, :].mean(axis=0)
    ax.plot(freqs_ref, mean_spec, label=label, color=color, linewidth=2)

ax.set_xlabel("Fréquence (Hz)")
ax.set_ylabel("Amplitude normalisée moyenne")
ax.set_title("Spectres moyens par espèce")
ax.set_xlim(FREQ_MIN, FREQ_MAX)
ax.legend()

plt.tight_layout()
plt.savefig(FIG / "spectres_moyens.png", dpi=150)
plt.close()

