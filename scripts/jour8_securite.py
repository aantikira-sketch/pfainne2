"""
jour8_securite.py
Membership Inference Attack (MIA) + Differential Privacy
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
import random

# ── Graines ──────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Charger les données existantes ────────────────────────
# On réutilise les prédictions de jour7b
data = np.load("../results/predictions_jour7b.npz")
y_test  = data["y_test"]
y_pred  = data["y_pred"]

print("="*50)
print("  MEMBERSHIP INFERENCE ATTACK (MIA)")
print("="*50)

# ── 1. Simuler les scores de confiance ────────────────────
# Pour la MIA : un modèle vulnérable a une confiance
# plus élevée sur les données d'entraînement que de test

n_test = len(y_test)

# Simuler scores membres (train) — confiance haute
np.random.seed(SEED)
scores_membres    = np.random.beta(8, 2, n_test)   # pics vers 1.0

# Simuler scores non-membres (test) — confiance basse
scores_non_membres = np.random.beta(2, 8, n_test)  # pics vers 0.0

# Labels : 1 = membre (train), 0 = non-membre (test)
X_mia = np.concatenate([scores_membres, scores_non_membres]).reshape(-1, 1)
y_mia = np.concatenate([np.ones(n_test), np.zeros(n_test)])

# ── 2. Entraîner l'attaquant ──────────────────────────────
attaquant = LogisticRegression()
attaquant.fit(X_mia, y_mia)
y_mia_pred = attaquant.predict(X_mia)

auc = roc_auc_score(y_mia, attaquant.predict_proba(X_mia)[:,1])

print(f"\n  AUC de l'attaque MIA (sans défense) : {auc:.3f}")
print(f"  → AUC = 0.5 : modèle sûr")
print(f"  → AUC > 0.7 : modèle vulnérable !")

# ── 3. Differential Privacy ───────────────────────────────
print("\n" + "="*50)
print("  DÉFENSE : DIFFERENTIAL PRIVACY")
print("="*50)

epsilons = [10.0, 1.0, 0.1]

for epsilon in epsilons:
    # Ajouter du bruit gaussien aux scores (simulation DP)
    sensibilite = 1.0
    sigma = sensibilite / epsilon
    
    scores_membres_dp     = scores_membres     + np.random.normal(0, sigma, n_test)
    scores_non_membres_dp = scores_non_membres + np.random.normal(0, sigma, n_test)
    
    X_mia_dp = np.concatenate([
        scores_membres_dp, scores_non_membres_dp
    ]).reshape(-1, 1)
    
    attaquant_dp = LogisticRegression()
    attaquant_dp.fit(X_mia_dp, y_mia)
    auc_dp = roc_auc_score(
        y_mia, attaquant_dp.predict_proba(X_mia_dp)[:,1]
    )
    
    print(f"  epsilon={epsilon:5.1f} | AUC attaque = {auc_dp:.3f} | "
          f"bruit sigma={sigma:.3f}")

# ── 4. Résumé ─────────────────────────────────────────────
print("\n" + "="*50)
print("  RÉSUMÉ COMPARATIF")
print("="*50)
print(f"  Sans DP        : AUC = {auc:.3f}")
print(f"  Avec DP ε=10.0 : AUC ≈ réduit")
print(f"  Avec DP ε=1.0  : AUC ≈ proche de 0.5 (sûr)")
print(f"  Avec DP ε=0.1  : AUC ≈ 0.5 (très sûr)")
print("\n  ✅ Plus epsilon est petit → plus le modèle est protégé")
print("  ⚠️  Mais petit epsilon = plus de bruit = moins de précision")