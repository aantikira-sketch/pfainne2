"""
=============================================================
 JOUR 6 — L'autoencodeur Bi-LSTM (vraie archi Manocha)
=============================================================

Au jour 5, ton autoencodeur traitait les 250 points EN VRAC
(couches Linear). Il a atteint 0.878. Bien.

Ici on respecte l'ordre TEMPOREL du battement : l'encodeur et le
decodeur sont des Bi-LSTM, qui lisent la sequence dans les deux sens
en gardant une memoire. C'est l'architecture de Manocha.

Choix pragmatiques (CPU) -- a noter dans ton journal de divergences :
  - on sous-echantillonne le battement de ~250 a ~90 points
    (le LSTM est lent : son temps croit avec la longueur de sequence)
  - autoencodeur sur 15 epoques, classifieur = petit reseau (jour 4b)

Pre-requis :
    pip install torch wfdb scikit-learn numpy

Lancer :
    python jour6_bilstm.py
"""

import time
import wfdb
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
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
# DONNEES — signal brut, puis SOUS-ECHANTILLONNAGE pour le LSTM
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

WIN = 0.35
STEP = 3   # on garde 1 point sur 3 -> ~250/3 ≈ 84 points (LSTM plus rapide)

print("Extraction + sous-echantillonnage des battements...")
X, y = [], []
for rec in RECORDS:
    record = wfdb.rdrecord(f'data/raw/mitbih/{rec}')
    ann    = wfdb.rdann(f'data/raw/mitbih/{rec}', 'atr')
    sig, fs = record.p_signal[:, 0], record.fs
    half = int(WIN * fs)
    for s, sym in zip(ann.sample, ann.symbol):
        if sym in sym2aami and s - half > 0 and s + half < len(sig):
            beat = sig[s - half : s + half : STEP]    # <-- sous-echantillonnage
            X.append(beat)
            y.append(cls2idx[sym2aami[sym]])

L = min(len(b) for b in X)
X = np.array([b[:L] for b in X], dtype=np.float32)
y = np.array(y)
print(f"Total : {len(y)} battements, longueur {L} points (apres sous-ech.)\n")

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

def normalize(a):
    m, s = a.mean(axis=1, keepdims=True), a.std(axis=1, keepdims=True) + 1e-6
    return ((a - m) / s).astype(np.float32)
Xtr_n, Xte_n = normalize(Xtr), normalize(Xte)

# Pour un LSTM, l'entree doit avoir la forme (batch, longueur, 1) :
# une sequence de L pas de temps, chacun de dimension 1 (signal mono-canal).
Xtr_seq = torch.tensor(Xtr_n).unsqueeze(-1)   # (N, L, 1)
Xte_seq = torch.tensor(Xte_n).unsqueeze(-1)


# =============================================================
# ETAGE 1 — AUTOENCODEUR Bi-LSTM
# =============================================================
class BiLSTMAutoEncodeur(nn.Module):
    def __init__(self, hidden=32, n_code=32, seq_len=L):
        super().__init__()
        self.seq_len = seq_len
        # ENCODEUR : Bi-LSTM lit la sequence dans les deux sens
        self.enc_lstm = nn.LSTM(input_size=1, hidden_size=hidden,
                                batch_first=True, bidirectional=True)
        # 2*hidden car bidirectionnel (un sens + l'autre concatenes)
        self.to_code = nn.Linear(2 * hidden, n_code)      # -> le goulot
        # DECODEUR : du code, on reconstruit la sequence
        self.from_code = nn.Linear(n_code, 2 * hidden)
        self.dec_lstm = nn.LSTM(input_size=2 * hidden, hidden_size=hidden,
                                batch_first=True, bidirectional=True)
        self.reconstruct = nn.Linear(2 * hidden, 1)

    def encode(self, x):
        out, _ = self.enc_lstm(x)          # (N, L, 2*hidden)
        pooled = out.mean(dim=1)           # moyenne sur le temps -> (N, 2*hidden)
        return self.to_code(pooled)        # (N, n_code) = le CODE

    def forward(self, x):
        code = self.encode(x)
        h = self.from_code(code).unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.dec_lstm(h)
        recon = self.reconstruct(out)      # (N, L, 1)
        return recon, code

ae = BiLSTMAutoEncodeur()
perte_recon = nn.MSELoss()
opt_ae = torch.optim.Adam(ae.parameters(), lr=0.005)

ds = TensorDataset(Xtr_seq)
dl = DataLoader(ds, batch_size=256, shuffle=True)

print("ETAGE 1 — entrainement de l'autoencodeur Bi-LSTM")
print("(le LSTM est lent : compte ~1-3 min/epoque sur CPU, c'est NORMAL)\n")
print("Epoque | Perte recon | temps")
print("-------|-------------|-------")
N_EP_AE = 15
for epoch in range(1, N_EP_AE + 1):
    ae.train(); t0 = time.time(); pertes = []
    for (xb,) in dl:
        opt_ae.zero_grad()
        recon, _ = ae(xb)
        perte = perte_recon(recon, xb)
        perte.backward(); opt_ae.step()
        pertes.append(perte.item())
    if epoch % 3 == 0 or epoch == 1:
        print(f"  {epoch:2d}   |   {np.mean(pertes):.4f}    | {time.time()-t0:.0f}s")
print("-> perte qui baisse = le Bi-LSTM apprend a resumer la SEQUENCE\n")


# =============================================================
# ETAGE 2 — ENCODER : sequence -> code de 32 features apprises
# =============================================================
ae.eval()
with torch.no_grad():
    Ctr = ae.encode(Xtr_seq).numpy()
    Cte = ae.encode(Xte_seq).numpy()
sc = StandardScaler().fit(Ctr)
Ctr_s = sc.transform(Ctr).astype(np.float32)
Cte_s = sc.transform(Cte).astype(np.float32)
print(f"ETAGE 2 — chaque battement = {Ctr.shape[1]} features TEMPORELLES apprises\n")


# =============================================================
# ETAGE 3 — CLASSIFIEUR : petit reseau (jour 4b), rapide
# =============================================================
class PetitReseau(nn.Module):
    def __init__(self, n_in=32, n_classes=5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 32), nn.ReLU(), nn.Linear(32, n_classes))
    def forward(self, x): return self.net(x)

clf = PetitReseau(n_in=Ctr_s.shape[1])
counts = np.bincount(ytr, minlength=5)
w = 1.0/np.sqrt(counts); w = w/w.sum()*5
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
opt = torch.optim.Adam(clf.parameters(), lr=0.005)

ds2 = TensorDataset(torch.tensor(Ctr_s), torch.tensor(ytr, dtype=torch.long))
dl2 = DataLoader(ds2, batch_size=128, shuffle=True)
print("ETAGE 3 — classification (petit reseau)...")
for epoch in range(40):
    clf.train()
    for xb, yb in dl2:
        opt.zero_grad(); loss_fn(clf(xb), yb).backward(); opt.step()
clf.eval()
with torch.no_grad():
    pred = clf(torch.tensor(Cte_s)).argmax(1).numpy()
mf1 = f1_score(yte, pred, average='macro')


# =============================================================
# COMPARAISON
# =============================================================
print("\n" + "=" * 56)
print("  RECAPITULATIF de toute ta progression (macro F1)")
print("-" * 56)
print(f"  Baseline RF (6 features)            jour 3  : 0.730")
print(f"  Petit reseau (6 features)           jour 4b : 0.674")
print(f"  Autoencodeur Linear + reseau        jour 5  : 0.878")
print(f"  Autoencodeur Bi-LSTM + reseau       jour 6  : {mf1:.3f}")
print("=" * 56)
print("\nDetail par classe — Bi-LSTM :")
print(classification_report(yte, pred,
      target_names=[idx2cls[i] for i in range(5)], zero_division=0))


# =============================================================
#  CE QUE TU DOIS OBSERVER
# =============================================================
# 1. Le Bi-LSTM respecte l'ORDRE du signal (P->QRS->T). Compare au
#    jour 5 (Linear, en vrac) : regarde surtout la PRECISION de F,
#    qui trainait a 0.40. L'info temporelle peut l'aider.
# 2. Le gain global peut etre modeste : tu partais deja de 0.878.
#    L'important est la FIDELITE a Manocha, pas seulement le chiffre.
# 3. Tu as desormais reproduit l'architecture exacte de Manocha :
#    autoencodeur Bi-LSTM -> features -> classifieur. WP1 est dans
#    tes mains de bout en bout.
#
#  Si tu veux la version 100% fidele : remettre le SVM en etage 3
#  (comme au jour 5) a la place du petit reseau.
