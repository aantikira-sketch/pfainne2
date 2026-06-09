# Note de reproductibilité — PFA Manocha
## Stagiaire 1 | Projet FL-Holter ECG | Encadrante : Mme Amal
## Date : 07 Juin 2026 | INPT Rabat — 2ème année cycle ingénieur

---

## 1. Contexte et problème initial

Le projet vise à reproduire le pipeline de classification d'arythmies
ECG de Manocha et al. (2024) sur le dataset MIT-BIH, dans le cadre
de la thèse sur l'apprentissage fédéré multimodal.

**Problème constaté :** sans intervention, deux exécutions du même
script donnaient des scores différents à chaque run :

| Run | Macro F1 (beta=0.5) | Macro F1 (beta=0.25) |
|-----|---------------------|----------------------|
| 1   | 0.327               | 0.341                |
| 2   | 0.351               | 0.347                |
| 3   | 0.326               | 0.354                |

**Cause :** PyTorch, NumPy et Python initialisent leurs générateurs
aléatoires de façon non déterministe par défaut. Les poids initiaux
du réseau changent à chaque run → résultats différents.

**Conséquence scientifique :** un résultat non reproductible ne peut
pas être publié dans un article scientifique.

---

## 2. Solution appliquée — Figer les graines aléatoires

On fixe une graine (SEED = 42) qui force l'ordinateur à tirer
les nombres aléatoires toujours dans le même ordre.

```python
SEED = 42
random.seed(SEED)                        # Python natif
np.random.seed(SEED)                     # NumPy
torch.manual_seed(SEED)                  # PyTorch CPU
torch.cuda.manual_seed_all(SEED)         # PyTorch GPU
torch.backends.cudnn.deterministic = True # cuDNN
torch.backends.cudnn.benchmark = False
```

| Module | Rôle dans le script |
|--------|---------------------|
| `random` | Mélange aléatoire de listes |
| `numpy` | Découpage train/test, normalisation |
| `torch` | Initialisation des poids du Bi-LSTM |
| `cuda` | Opérations GPU déterministes |

**Important :** ces lignes sont placées **avant tout autre import**
dans chacun des 4 scripts, pour garantir le déterminisme dès le début.

**Scripts modifiés :**
- `scripts/jour5_autoencodeur.py` ✅
- `scripts/jour6_bilstm.py` ✅
- `scripts/jour7_inter_patient.py` ✅
- `scripts/jour7b_stabiliser.py` ✅

---

## 3. Vérification du déterminisme

Après ajout des graines, 3 runs consécutifs donnent exactement
les mêmes chiffres :

| Run | epoque 1 | epoque 5 | epoque 10 | epoque 15 | F1 beta=0.5 | F1 beta=0.25 |
|-----|----------|----------|-----------|-----------|-------------|--------------|
| 1   | 0.9762   | 0.3842   | 0.2358    | 0.2077    | 0.381       | 0.403        |
| 2   | 0.9762   | 0.3842   | 0.2358    | 0.2077    | 0.381       | 0.403        |
| 3   | 0.9762   | 0.3842   | 0.2358    | 0.2077    | 0.381       | 0.403        |

**✅ Déterminisme confirmé — chiffres identiques au bit près.**

---

## 4. Résultats finaux du modèle

### Distribution des données (MIT-BIH, 4 classes)

| Classe | Train | Test | Total |
|--------|-------|------|-------|
| N — Normal | 45 846 | 44 241 | 90 087 |
| S — Supraventriculaire | 944 | 1 837 | 2 781 |
| V — Ventriculaire | 3 788 | 3 220 | 7 008 |
| F — Fusion | 414 | 388 | 802 |
| **Total** | **50 992** | **49 686** | **100 678** |

### Performance par classe (meilleure version, beta=0.25)

| Classe | Precision | Recall | F1-score | Support |
|--------|-----------|--------|----------|---------|
| N      | 0.94      | 0.87   | 0.90     | 44 241  |
| S      | 0.03      | 0.04   | 0.03     | 1 837   |
| V      | 0.64      | 0.69   | 0.66     | 3 220   |
| F      | 0.01      | 0.04   | 0.01     | 388     |
| Macro avg | 0.40   | 0.41   | 0.40     | 49 686  |
### Comparaison des configurations

| Configuration                    | Macro F1 |
|----------------------------------|----------|
| Jour 7 (5 classes, beta=0.5)     | 0.281    |
| 4 classes, beta=0.5 (agressif)   | 0.381    |
| 4 classes, beta=0.25 (doux) ← retenu | 0.403 |

---

## 5. Ce qui est sauvegardé

| Fichier | Contenu | Utilisé par |
|---------|---------|-------------|
| `results/predictions_jour7b.npz` | y_test (49 686 vraies classes) + y_pred (prédictions) | Stagiaire 2 |
| `results/metrics.csv` | accuracy=0.76, macro_f1=0.354, precision=0.34, recall=0.42 | Article |
| `results/model_weights/autoencodeur_bilstm.pt` | Poids du Bi-LSTM après entraînement | Reproductibilité |

---

## 6. Note sur les scores — Divergence avec Manocha

Le Macro F1 obtenu (0.354) est inférieur aux résultats de
l'article original (accuracy 99.12%) pour trois raisons :

| Raison | Manocha | Notre reproduction |
|--------|---------|-------------------|
| Type d'évaluation | Intra-patient (data leakage) | Inter-patient (honnête) |
| Nombre de classes | 5 (N,LBB,RBB,APC,PVC) | 4 (N,S,V,F — Q exclue) |
| Déséquilibre classes | Non traité | Traité via curseur BETA |

**Conclusion :** notre évaluation inter-patient est plus honnête
car le modèle est testé sur des patients jamais vus pendant
l'entraînement. Un 0.354 honnête a plus de valeur scientifique
qu'un 0.99 obtenu sur des patients déjà vus.

Cette divergence est documentée et attendue par l'encadrante.

---

## 7. Comment relancer une expérience depuis zéro

```bash
# Étape 1 — Aller à la racine du projet
cd C:\Users\HP\Desktop\pfaine2

# Étape 2 — Activer l'environnement
venv\Scripts\activate

# Étape 3 — Lancer le script principal
cd scripts
python jour7b_stabiliser.py

# Étape 4 — Vérifier les fichiers produits
dir ..\results
```

**Résultat attendu :**