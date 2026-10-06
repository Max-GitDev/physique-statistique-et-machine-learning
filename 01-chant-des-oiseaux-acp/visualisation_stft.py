#!/usr/bin/env python3
"""
Le chant des oiseaux - Partie 1 : visualisation par transformée de Fourier à court terme (STFT).

Implémentation de la STFT avec NumPy (recouvrement de 50 %), étude du compromis
temps-fréquence selon la taille de fenêtre N_s, comparaison avec scipy.signal.stft
et spectrogrammes de quatre espèces.

Auteurs : Maxime Rimbaud et Romain Bertin - ESPCI Paris - PSL (mars 2026).
Usage : python visualisation_stft.py [dossier_des_enregistrements]
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


# QUESTION 1 — Chargement des fichiers .wav


def load_wav(path):
    """Charge un fichier .wav et renvoie (fs, signal mono float)."""
    fs, data = wavfile.read(path)
    if data.ndim > 1:          # stéréo → canal gauche uniquement
        data = data[:, 0]
    return fs, data.astype(float)

fs_h, hibou  = load_wav(DATA / "part1" / "hibou_grand_duc.wav")
fs_a, aigle  = load_wav(DATA / "part1" / "aigle_royal.wav")
fs_l, loriot = load_wav(DATA / "part1" / "loriot.wav")
fs_r, rossig = load_wav(DATA / "part1" / "rossignol.wav")

print(f"Hibou    : fs={fs_h} Hz, durée={len(hibou)/fs_h:.2f}s")
print(f"Aigle    : fs={fs_a} Hz, durée={len(aigle)/fs_a:.2f}s")
print(f"Loriot   : fs={fs_l} Hz, durée={len(loriot)/fs_l:.2f}s")
print(f"Rossignol: fs={fs_r} Hz, durée={len(rossig)/fs_r:.2f}s")


# QUESTION 2 — Implémentation de la STFT


def stft(signal, fs, N_s):
    """
    Short-Time Fourier Transform (STFT) avec recouvrement de 50 %.
    """
    hop    = N_s // 2                           # pas = 50 % de recouvrement
    N      = len(signal)
    nb_seg = (N - N_s) // hop + 1              # nb de segments complets

    # Matrice de sortie : N_s lignes (fréquences), nb_seg colonnes (segments)
    S = np.zeros((N_s, nb_seg), dtype=complex)

    for j in range(nb_seg):
        start    = j * hop                      # début : j * Ns/2
        segment  = signal[start : start + N_s]  # extraction du segment
        S[:, j]  = np.fft.fft(segment)         # TF du segment j

    # Temps : milieu de la fenêtre j = (j·hop + Ns/2) / fs
    times = np.array([(j * hop + N_s / 2) / fs for j in range(nb_seg)])

    # Fréquences associées à la FFT de taille N_s
    freqs = np.fft.fftfreq(N_s, d=1.0 / fs)

    return S, times, freqs


# QUESTION 3 — Spectrogramme du hibou grand-duc, Ns = 2^10


N_s = 2**10   # 1024 points

S_h, times_h, freqs_h = stft(hibou, fs_h, N_s)

# Paramètres d'affichage
t_min,    t_max    = 12.0,  13.75   # fenêtre temporelle (secondes)
freq_min, freq_max = 0,     10000   # gamme fréquentielle (Hz)

# Sélection des indices
tm = (times_h >= t_min) & (times_h <= t_max)
fm = (freqs_h >= freq_min) & (freqs_h <= freq_max)

S_disp = np.abs(S_h[fm, :][:, tm])
v_max  = np.percentile(S_disp, 99)   # seuillage à 99e percentile pour le contraste

fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(S_disp, origin="lower",
               extent=(t_min, t_max, freq_min, freq_max),
               aspect="auto", vmax=v_max)
plt.colorbar(im, ax=ax, label="Amplitude |S|")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Frequency [Hz]")
ax.set_title("Grand-duc d'Europe (Bubo bubo) — $N_s = 2^{10}$")

plt.tight_layout()
plt.savefig(FIG / "spectrogramme_grand_duc.png", dpi=150)
plt.close()


# QUESTION 4 — Effet de Ns


fig, axes = plt.subplots(2, 2, figsize=(14, 8))

for ax, ns in zip(axes.flat, [64, 256, 1024, 4096]):
    S_, t_, f_ = stft(hibou, fs_h, ns)
    tm_ = (t_ >= t_min)  & (t_ <= t_max)
    fm_ = (f_ >= 0)      & (f_ <= 10000)
    S_d = np.abs(S_[fm_, :][:, tm_])
    vmax_ = np.percentile(S_d, 99)
    im = ax.imshow(S_d, origin="lower",
                   extent=(t_min, t_max, 0, 10000),
                   aspect="auto", vmax=vmax_)
    plt.colorbar(im, ax=ax)
    ax.set_title(f"$N_s = {ns}$")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")

plt.suptitle("Effet de $N_s$ sur la STFT — Grand-duc d'Europe", fontsize=13)

plt.tight_layout()
plt.savefig(FIG / "effet_taille_fenetre.png", dpi=150)
plt.close()


# QUESTION 5 — Comparaison avec scipy.signal.stft


# Notre implémentation
S_maison, times_maison, freqs_maison = stft(hibou, fs_h, N_s)

# Implémentation scipy 
f_scipy, t_scipy, Z_scipy = scipy_signal.stft(
    hibou, fs=fs_h, nperseg=N_s, noverlap=N_s // 2
)

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, (label, S_, t_, f_) in zip(axes, [
    ("Implémentation maison",  S_maison, times_maison, freqs_maison),
    ("scipy.signal.stft",      Z_scipy,  t_scipy,      f_scipy),
]):
    tm_ = (t_ >= t_min) & (t_ <= t_max)
    fm_ = (f_ >= 0)     & (f_ <= 10000)
    S_d = np.abs(S_[fm_, :][:, tm_])
    vmax_ = np.percentile(S_d, 99)
    im = ax.imshow(S_d, origin="lower",
                   extent=(t_min, t_max, 0, 10000),
                   aspect="auto", vmax=vmax_)
    plt.colorbar(im, ax=ax, label="|S|")
    ax.set_title(label)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")

plt.suptitle("Comparaison implémentation maison vs scipy — Grand-duc", fontsize=12)

plt.tight_layout()
plt.savefig(FIG / "comparaison_scipy.png", dpi=150)
plt.close()


# QUESTION 6 — Spectrogrammes des 4 oiseaux

# Choix des fenêtres temporelles : on cible des séquences de chant actif
# t_min/t_max choisis après inspection visuelle du signal complet.
# v_max ajusté au 99e percentile de chaque spectrogramme pour maximiser le contraste sans saturation.

birds = [
    # (titre, signal, fs, t_min, t_max)
    ("Grand-duc d'Europe\n(Bubo bubo)",      hibou,  fs_h, 12.0, 13.75),
    ("Aigle royal\n(Aquila chrysaetos)",     aigle,  fs_a,  1.0,  3.5),
    ("Loriot d'Europe\n(Oriolus oriolus)",   loriot, fs_l,  1.0,  3.0),
    ("Rossignol philomène\n(luscinia meg.)", rossig, fs_r,  1.0,  3.5),
]

fig, axes = plt.subplots(2, 2, figsize=(15, 9))

for ax, (title, sig, fs, tmin, tmax) in zip(axes.flat, birds):
    S_, t_, f_ = stft(sig, fs, N_s)
    tm_ = (t_ >= tmin) & (t_ <= tmax)
    fm_ = (f_ >= 0)    & (f_ <= 10000)
    S_d = np.abs(S_[fm_, :][:, tm_])
    vmax_ = np.percentile(S_d, 99)
    im = ax.imshow(S_d, origin="lower",
                   extent=(tmin, tmax, 0, 10000),
                   aspect="auto", vmax=vmax_)
    plt.colorbar(im, ax=ax, label="|S|")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")

plt.suptitle("Spectrogrammes des 4 oiseaux — $N_s = 2^{10}$", fontsize=13)

plt.tight_layout()
plt.savefig(FIG / "spectrogrammes_4_oiseaux.png", dpi=150)
plt.close()
