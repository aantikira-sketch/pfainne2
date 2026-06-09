"""
=============================================================
 JOUR 5 — L'autoencodeur (enfin !) facon Manocha
=============================================================

L'aboutissement de la serie. On arrete de donner 6 features pauvres
au classifieur. A la place :

  ETAGE 1 : un autoencodeur apprend a RESUMER le signal BRUT (250 points)
            en un code de 32 nombres, en se reconstruisant lui-meme.
  ETAGE 2 : on gele l'encodeur et on transforme chaque battement en code.
  ETAGE 3 : on classe ces 32 features apprises avec DEUX classifieurs
            (SVM facon Manocha + petit reseau du jour 4b) et on compare.

Question de recherche : de meilleures features (apprises) font-elles
enfin decoller le macro F1, surtout le recall des classes rares ?

Pre-requis :
    pip install torch wfdb scikit-learn numpy

Lancer :
    python jour5_autoencodeur.py
"""

import wfdb
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
import random
import os

# ═══════════════════════════════════════════════════
# REPRODUCTIBILITÉ — Graines aléatoires figées
# ═══════════════════════════════════════════════════
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# ═══════════════════════════════════════════════════

# =============================================================
# DONNEES — cette fois on garde le SIGNAL BRUT, pas les 6 features
# =============================================================
wfdb.dl_database('mitdb', dl_dir='data/raw/mitbih')

AAMI = {'N': ['N','L','R','e','j'], 'S': ['A','a','J','S'],
        'V': ['V','E'], 'F': ['F'], 'Q': ['/','f','Q']}
sym2aami = {s: c for c, ss in AAMI.items() for s in ss}
cls2idx  = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}
idx2cls  = {v: k for k, v in cls2idx.items()}
RECORDS = ['100','101','102','103','104','105','106','107','108','109',
           '111','112','113','114','115','116','117','118','119','121',
           '122','123','124','200','201','202','203','205','207','208',
           '209','210','212','213','214','215','217','219','220','221',
           '222','223','228','230','231','232','233','234']

WIN = 0.35   # demi-fenetre en secondes -> a 360 Hz : 0.35*360*2 = 252 points

print("Extraction des battements BRUTS (250 points chacun)...")
X, y = [], []
for rec in RECORDS:
    record = wfdb.rdrecord(f'data/raw/mitbih/{rec}')
    ann    = wfdb.rdann(f'data/raw/mitbih/{rec}', 'atr')
    sig, fs = record.p_signal[:, 0], record.fs
    half = int(WIN * fs)
    for s, sym in zip(ann.sample, ann.symbol):
        if sym in sym2aami and s - half > 0 and s + half < len(sig):
            beat = sig[s - half : s + half]
            X.append(beat)                       # <-- le SIGNAL BRUT entier
            y.append(cls2idx[sym2aami[sym]])

# harmoniser la longueur (securite si quelques battements different d'1 point)
L = min(len(b) for b in X)
X = np.array([b[:L] for b in X], dtype=np.float32)
y = np.array(y)
print(f"Total : {len(y)} battements, chacun de longueur {L} points\n")

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# normaliser chaque battement (moyenne 0, ecart-type 1) : aide l'autoencodeur
def normalize(a):
    m, s = a.mean(axis=1, keepdims=True), a.std(axis=1, keepdims=True) + 1e-6
    return ((a - m) / s).astype(np.float32)
Xtr_n, Xte_n = normalize(Xtr), normalize(Xte)


# =============================================================
# ETAGE 1 — L'AUTOENCODEUR
# =============================================================
# encodeur : 250 -> 64 -> 32 (le goulot)
# decodeur : 32 -> 64 -> 250 (reconstruction)
class AutoEncodeur(nn.Module):
    def __init__(self, n_in, n_code=32):
        super().__init__()
        self.encodeur = nn.Sequential(
            nn.Linear(n_in, 64), nn.ReLU(),
            nn.Linear(64, n_code), nn.ReLU(),     # <-- le CODE (goulot)
        )
        self.decodeur = nn.Sequential(
            nn.Linear(n_code, 64), nn.ReLU(),
            nn.Linear(64, n_in),                  # <-- reconstruction
        )
    def forward(self, x):
        code = self.encodeur(x)
        recon = self.decodeur(code)
        return recon, code

ae = AutoEncodeur(n_in=L, n_code=32)

# La perte de RECONSTRUCTION : ecart entre l'original et la copie (MSE).
# Note : AUCUNE etiquette ici. C'est l'apprentissage non supervise.
perte_recon = nn.MSELoss()
opt_ae = torch.optim.Adam(ae.parameters(), lr=0.001)

ds = TensorDataset(torch.tensor(Xtr_n))
dl = DataLoader(ds, batch_size=128, shuffle=True)

print("ETAGE 1 — entrainement de l'autoencodeur (se reconstruire)")
print("Epoque | Perte de reconstruction")
print("-------|------------------------")
for epoch in range(1, 21):
    ae.train()
    pertes = []
    for (xb,) in dl:
        opt_ae.zero_grad()
        recon, _ = ae(xb)
        perte = perte_recon(recon, xb)     # la copie doit ressembler a l'original
        perte.backward()
        opt_ae.step()
        pertes.append(perte.item())
    if epoch % 4 == 0 or epoch == 1:
        print(f"  {epoch:2d}   |        {np.mean(pertes):.4f}")
print("-> la perte baisse = la copie ressemble de plus en plus a l'original")
print("   = le code de 32 nombres capture bien la forme du battement\n")


# =============================================================
# ETAGE 2 — ENCODER : battement (250) -> code (32)
# =============================================================
ae.eval()
with torch.no_grad():
    Ctr = ae.encodeur(torch.tensor(Xtr_n)).numpy()   # features apprises (train)
    Cte = ae.encodeur(torch.tensor(Xte_n)).numpy()   # features apprises (test)
print(f"ETAGE 2 — chaque battement est maintenant {Ctr.shape[1]} features apprises\n")

# on standardise les codes avant de classer
sc = StandardScaler().fit(Ctr)
Ctr_s, Cte_s = sc.transform(Ctr).astype(np.float32), sc.transform(Cte).astype(np.float32)


# =============================================================
# ETAGE 3a — CLASSIFIEUR 1 : SVM (facon Manocha)
# =============================================================
print("ETAGE 3a — SVM sur les features apprises (facon Manocha)...")
svm = SVC(C=1.0, kernel='rbf', gamma='scale', class_weight='balanced')
svm.fit(Ctr_s, ytr)
pred_svm = svm.predict(Cte_s)
f1_svm = f1_score(yte, pred_svm, average='macro')
print(f"   macro F1 (SVM)   : {f1_svm:.3f}\n")


# =============================================================
# ETAGE 3b — CLASSIFIEUR 2 : petit reseau (jour 4b)
# =============================================================
print("ETAGE 3b — petit reseau sur les MEMES features apprises...")
class PetitReseau(nn.Module):
    def __init__(self, n_in=32, n_classes=5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 32), nn.ReLU(), nn.Linear(32, n_classes))
    def forward(self, x): return self.net(x)

clf = PetitReseau(n_in=Ctr_s.shape[1])
counts = np.bincount(ytr, minlength=5)
w = 1.0/np.sqrt(counts); w = w/w.sum()*5            # poids ADOUCI (lecon jour 4b)
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
opt = torch.optim.Adam(clf.parameters(), lr=0.005)

ds2 = TensorDataset(torch.tensor(Ctr_s), torch.tensor(ytr, dtype=torch.long))
dl2 = DataLoader(ds2, batch_size=128, shuffle=True)
for epoch in range(30):
    clf.train()
    for xb, yb in dl2:
        opt.zero_grad(); loss_fn(clf(xb), yb).backward(); opt.step()
clf.eval()
with torch.no_grad():
    pred_net = clf(torch.tensor(Cte_s)).argmax(1).numpy()
f1_net = f1_score(yte, pred_net, average='macro')
print(f"   macro F1 (reseau): {f1_net:.3f}\n")


# =============================================================
# COMPARAISON FINALE
# =============================================================
print("=" * 54)
print("  COMPARAISON  (toutes sur les MEMES 32 features apprises)")
print("-" * 54)
print(f"  Baseline RF (6 features faites main, jour 3) : 0.730")
print(f"  Petit reseau (6 features, jour 4b)           : 0.674")
print(f"  Autoencodeur + SVM    (facon Manocha)        : {f1_svm:.3f}")
print(f"  Autoencodeur + reseau                        : {f1_net:.3f}")
print("=" * 54)

print("\nDetail par classe — Autoencodeur + SVM :")
print(classification_report(yte, pred_svm,
      target_names=[idx2cls[i] for i in range(5)], zero_division=0))


# =============================================================
#  CE QUE TU DOIS OBSERVER ET RETENIR
# =============================================================
# 1. ETAGE 1 : la perte de reconstruction baisse SANS etiquettes.
#    C'est l'apprentissage non supervise : le coeur de l'idee.
# 2. SVM vs reseau sur les MEMES features : s'ils donnent des scores
#    proches, tu as prouve que ce sont les FEATURES qui comptent, pas
#    tellement le classifieur. C'est un vrai resultat de recherche.
# 3. Compare au 0.730. Honnetement, le gain peut etre modeste sur des
#    battements aussi courts : l'important est la DEMARCHE, pas le chiffre.
#    Si le recall de S et F monte vs jour 4b, l'autoencodeur a aide.
#
# TU AS MAINTENANT REPRODUIT LA LOGIQUE DE MANOCHA DE BOUT EN BOUT :
#   autoencodeur -> features apprises -> SVM -> classes.
# C'est le coeur de WP1. La suite (vrai Manocha fidele, puis FL, puis
# cross-lead) ne fait que raffiner cette meme chaine.
