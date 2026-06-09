"""
=============================================================
 figures_exploratoires.py — Stagiaire 2
 PFA INPT 2e année — Projet FL-Holter ECG
 Encadrante : Amal
=============================================================
 Generates all figures for the Results section.
 Run : python figures_exploratoires.py

 Generated figures :
   fig1_distribution_classes.png   — class imbalance
   fig2_battements_types.png       — one sample beat per class
   fig3_signal_annote.png          — annotated ECG signal
   fig4_confusion_matrix.png       — confusion matrix
   fig5_accuracy_vs_f1.png         — accuracy vs macro F1
=============================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import wfdb
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                              accuracy_score, f1_score)

# ------------------------------------------------------------------
# GLOBAL CONFIGURATION
# ------------------------------------------------------------------
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

CLASS_NAMES = ['N', 'S', 'V', 'F']   # 4 AAMI classes (Q excluded)

# Mapping MIT-BIH symbols -> AAMI classes (identical to S1 scripts)
AAMI = {
    'N': ['N', 'L', 'R', 'e', 'j'],
    'S': ['A', 'a', 'J', 'S'],
    'V': ['V', 'E'],
    'F': ['F'],
}
sym2aami = {s: c for c, ss in AAMI.items() for s in ss}


# ==================================================================
# STEP 0 — Check that S1 file is present
# ==================================================================
def verifier_fichier_s1():
    """
    Checks that results/predictions_jour7b.npz exists.
    This file is produced by Stagiaire 1 (jour7b_stabiliser.py).
    If missing: displays a clear message and stops the script.

    While waiting for the real file (days 1-3), manually create
    a fake file with create_fake_file.py (separate script).
    """
    chemin = "results/predictions_jour7b.npz"
    if not os.path.exists(chemin):
        print("=" * 55)
        print("  ERROR: file not found:")
        print(f"  {chemin}")
        print()
        print("  This file is produced by Stagiaire 1.")
        print("  Ask them to run jour7b_stabiliser.py")
        print("  OR use create_fake_file.py for testing.")
        print("=" * 55)
        exit(1)
    print(f"-> File found: {chemin}")


# ==================================================================
# FIGURE 1 — Class distribution (imbalance)
# ==================================================================
def fig1_distribution():
    classes = ['N\n(Normal)', 'S\n(Supra-V)', 'V\n(Ventricular)', 'F\n(Fusion)']
    colors  = ['steelblue', 'orange', 'green', 'red']

    # Lire les vrais effectifs depuis le fichier de S1
    data = np.load("results/predictions_jour7b.npz")
    if 'class_counts' in data.files:
        counts = data['class_counts'].tolist()
        print("   Effectifs lus depuis le fichier de S1")
    else:
        counts = [89839, 3433, 7374, 150]
        print(" Effectifs par défaut — S1 doit ajouter class_counts")

    total = sum(counts)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(classes, counts, color=colors, edgecolor='white')

    ax.set_title("AAMI Class Distribution — MIT-BIH\n"
                 "Massive imbalance: N accounts for 88% of all beats",
                 fontsize=12)
    ax.set_ylabel("Number of beats")

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 800,
                f'{count:,}\n({count/total*100:.1f}%)',
                ha='center', fontsize=9)

    ax.set_ylim(0, max(counts) * 1.2)
    plt.tight_layout()
    plt.savefig("figures/fig1_distribution_classes.png", dpi=150)
    plt.close()
    print("fig1 saved: figures/fig1_distribution_classes.png")


# ==================================================================
# FIGURE 2 — One sample beat per class
# ==================================================================
def fig2_battements_types():
    """
    Displays one example beat for each class N, S, V, F
    extracted directly from the MIT-BIH database.
    Records chosen: those that reliably contain each class.
    """
    # Records that contain each class with certainty
    sources = {
        'N': ('100', 'N'),   # Normal — very common
        'S': ('209', 'A'),   # Supraventricular — record 209 has many
        'V': ('106', 'V'),   # Ventricular — record 106 has many
        'F': ('214', 'F'),   # Fusion — record 214 has some
    }

    WIN = 0.35   # half-window in seconds (identical to S1 scripts)

    exemples = {}
    for classe, (rec, sym_cible) in sources.items():
        record = wfdb.rdrecord(f'data/raw/mitbih/{rec}')
        ann    = wfdb.rdann(f'data/raw/mitbih/{rec}', 'atr')
        sig, fs = record.p_signal[:, 0], record.fs
        half = int(WIN * fs)
        for s, sym in zip(ann.sample, ann.symbol):
            if sym == sym_cible and s - half > 0 and s + half < len(sig):
                exemples[classe] = sig[s - half: s + half]
                break   # take the first one found

    fig, axes = plt.subplots(1, 4, figsize=(14, 3), sharey=True)
    colors = {'N': 'steelblue', 'S': 'orange', 'V': 'green', 'F': 'red'}
    labels = {'N': 'N — Normal', 'S': 'S — Supra-V', 'V': 'V — Ventricular', 'F': 'F — Fusion'}

    for ax, classe in zip(axes, ['N', 'S', 'V', 'F']):
        beat = exemples.get(classe)
        if beat is not None:
            t = np.arange(len(beat))
            ax.plot(t, beat, color=colors[classe], linewidth=1.5)
        ax.set_title(labels[classe], fontsize=10)
        ax.set_xlabel("Samples")
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')

    axes[0].set_ylabel("Amplitude (mV)")
    fig.suptitle("Typical beat morphology for the 4 classes — MIT-BIH", fontsize=12)
    plt.tight_layout()
    plt.savefig("figures/fig2_battements_types.png", dpi=150)
    plt.close()
    print("fig2 saved: figures/fig2_battements_types.png")


# ==================================================================
# FIGURE 3 — Annotated ECG signal over a few seconds
# ==================================================================
def fig3_signal_annote():
    """
    Displays the raw ECG signal from record 100
    with beat-type annotations overlaid.
    """
    record = wfdb.rdrecord('data/raw/mitbih/100', sampto=3600)   # ~10 seconds
    ann    = wfdb.rdann('data/raw/mitbih/100', 'atr', sampto=3600)
    signal = record.p_signal[:, 0]
    t      = [i / record.fs for i in range(len(signal))]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, signal, color='black', linewidth=0.7, label='ECG MLII')

    for s, sym in zip(ann.sample, ann.symbol):
        ax.axvline(x=s / record.fs, color='red', alpha=0.4, linewidth=0.6)
        ax.text(s / record.fs, signal[s] + 0.05, sym,
                fontsize=6, color='red', ha='center')

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (mV)")
    ax.set_title("ECG Signal MIT-BIH (record 100) — First 10 seconds\n"
                 "Red annotations: detected beat type")
    plt.tight_layout()
    plt.savefig("figures/fig3_signal_annote.png", dpi=150)
    plt.close()
    print("fig3 saved: figures/fig3_signal_annote.png")


# ==================================================================
# FIGURE 4 — Confusion matrix
# ==================================================================
def fig4_confusion_matrix():
    """
    Reads results/predictions_jour7b.npz and displays the confusion matrix.
    Works with both the fake file (days 1-3) and the real S1 file (day 4+).
    """
    data   = np.load("results/predictions_jour7b.npz")
    y_test = data['y_test']
    y_pred = data['y_pred']

    # Detect whether this is the fake file or the real one
    est_faux = (len(y_test) == 500)
    titre = ("Confusion Matrix\n⚠ DUMMY DATA — to be replaced on day 4"
             if est_faux else
             "Confusion Matrix — Real results (Bi-LSTM-AE + classifier)")

    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap='Blues', colorbar=True)
    ax.set_title(titre, fontsize=10)
    plt.tight_layout()
    plt.savefig("figures/fig4_confusion_matrix.png", dpi=150)
    plt.close()
    print("fig4 saved: figures/fig4_confusion_matrix.png")


# ==================================================================
# FIGURE 5 — Accuracy vs Macro F1
# ==================================================================
def fig5_accuracy_vs_f1():
    """
    Compares accuracy and macro F1 to show that accuracy
    is misleading on imbalanced data.
    The dashed line at 88% = score of a 'always predict N' model.
    """
    data   = np.load("results/predictions_jour7b.npz")
    y_test = data['y_test']
    y_pred = data['y_pred']

    acc      = accuracy_score(y_test, y_pred) * 100
    macro_f1 = f1_score(y_test, y_pred, average='macro') * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(['Accuracy', 'Macro F1'],
                  [acc, macro_f1],
                  color=['steelblue', 'tomato'],
                  width=0.4, edgecolor='white')

    for bar, val in zip(bars, [acc, macro_f1]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontsize=13, fontweight='bold')

    ax.set_ylim(0, 115)
    ax.set_ylabel("Score (%)")
    ax.set_title("Accuracy vs Macro F1\n"
                 "Accuracy overestimates performance on imbalanced data\n"
                 "(always predicting N already gives ~88% accuracy!)",
                 fontsize=10)
    ax.axhline(y=88, color='gray', linestyle='--', linewidth=1.2,
               label="Baseline 'always N' ≈ 88%")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("figures/fig5_accuracy_vs_f1.png", dpi=150)
    plt.close()
    print("fig5 saved: figures/fig5_accuracy_vs_f1.png")


# ==================================================================
# MAIN — run everything at once
# ==================================================================
if __name__ == "__main__":
    print("=" * 55)
    print("  figures_exploratoires.py — Stagiaire 2")
    print("=" * 55)

    print("\n[0] Checking predictions file (produced by S1)...")
    verifier_fichier_s1()

    print("\n[A] Figures on RAW DATA")
    print("--- fig1: class distribution ---")
    fig1_distribution()

    print("--- fig2: sample beats ---")
    # Download MIT-BIH data if missing
    if not os.path.exists("data/raw/mitbih/100.dat"):
        print("    Downloading MIT-BIH (one time only)...")
        wfdb.dl_database('mitdb', dl_dir='data/raw/mitbih')
    fig2_battements_types()

    print("--- fig3: annotated ECG signal ---")
    fig3_signal_annote()

    print("\n[B] Figures on RESULTS (via interface contract)")
    print("--- fig4: confusion matrix ---")
    fig4_confusion_matrix()

    print("--- fig5: accuracy vs macro F1 ---")
    fig5_accuracy_vs_f1()

    print("\n" + "=" * 55)
    print("  ALL FIGURES GENERATED in figures/")
    print("  To regenerate: python figures_exploratoires.py")
    print("=" * 55)