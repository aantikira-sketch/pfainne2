"""
=============================================================
 JOUR 7 — Evaluation HONNETE : split inter-patient (par record)
=============================================================

Jusqu'ici on melangeait les battements de tous les patients avant de
couper train/test. Probleme : un meme patient avait des battements des
DEUX cotes -> le modele reconnaissait des patients deja vus -> score
gonfle. C'est une FUITE DE DONNEES (data leakage).

Correction : on coupe AU NIVEAU DU RECORD. Certains records entierement
en entrainement, les autres entierement en test. Aucun chevauchement.
C'est l'evaluation inter-patient (ici : inter-record).

Attendu : le macro F1 va sans doute BAISSER vs jour 6 (~0.87).
Ce n'est pas une regression : c'est la VERITE. Un 0.80 honnete vaut
mieux qu'un 0.87 trompeur.

Pre-requis : pip install torch wfdb scikit-learn numpy
Lancer    : python jour7_inter_patient.py
"""

import time
import wfdb
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
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

wfdb.dl_database('mitdb', dl_dir='data/raw/mitbih')

AAMI = {'N': ['N','L','R','e','j'], 'S': ['A','a','J','S'],
        'V': ['V','E'], 'F': ['F'], 'Q': ['/','f','Q']}
sym2aami = {s: c for c, ss in AAMI.items() for s in ss}
cls2idx  = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}
idx2cls  = {v: k for k, v in cls2idx.items()}

# -------------------------------------------------------------
# LE CHANGEMENT CLE — on decide quels RECORDS vont ou, AVANT tout
# -------------------------------------------------------------
# Decoupage inspire du standard DS1/DS2 de la litterature, choisi pour
# que les 5 classes soient presentes des deux cotes. Aucun record
# n'est dans les deux listes -> aucune fuite de patient.
TRAIN_RECORDS = ['101','106','108','109','112','114','115','116','118','119',
                 '122','124','201','203','205','207','208','209','215','220',
                 '223','230']
TEST_RECORDS  = ['100','103','105','111','113','117','121','123','200','202',
                 '210','212','213','214','219','221','222','228','231','232',
                 '233','234']

# verifier qu'aucun record n'est dans les deux camps
assert set(TRAIN_RECORDS).isdisjoint(TEST_RECORDS), "Fuite ! un record est dans les deux."

WIN, STEP = 0.35, 3

def charger(records):
    """Extrait les battements sous-echantillonnes d'une liste de records."""
    X, y = [], []
    for rec in records:
        record = wfdb.rdrecord(f'data/raw/mitbih/{rec}')
        ann    = wfdb.rdann(f'data/raw/mitbih/{rec}', 'atr')
        sig, fs = record.p_signal[:, 0], record.fs
        half = int(WIN * fs)
        for s, sym in zip(ann.sample, ann.symbol):
            if sym in sym2aami and s - half > 0 and s + half < len(sig):
                X.append(sig[s - half : s + half : STEP])
                y.append(cls2idx[sym2aami[sym]])
    return X, y

print("Chargement TRAIN (records d'entrainement)...")
Xtr_l, ytr = charger(TRAIN_RECORDS)
print("Chargement TEST  (records JAMAIS vus)...")
Xte_l, yte = charger(TEST_RECORDS)

L = min(min(len(b) for b in Xtr_l), min(len(b) for b in Xte_l))
Xtr = np.array([b[:L] for b in Xtr_l], dtype=np.float32)
Xte = np.array([b[:L] for b in Xte_l], dtype=np.float32)
ytr, yte = np.array(ytr), np.array(yte)

# VERIFICATION : les 5 classes sont-elles presentes des deux cotes ?
print("\nRepartition des classes (verifie qu'aucune n'est absente d'un cote) :")
print("classe | train  | test")
for i, c in idx2cls.items():
    print(f"   {c}   | {int((ytr==i).sum()):6d} | {int((yte==i).sum()):5d}")
print(f"\nTRAIN : {len(ytr)} battements ({len(TRAIN_RECORDS)} records)")
print(f"TEST  : {len(yte)} battements ({len(TEST_RECORDS)} records, jamais vus)\n")

def normalize(a):
    m, s = a.mean(axis=1, keepdims=True), a.std(axis=1, keepdims=True) + 1e-6
    return ((a - m) / s).astype(np.float32)
Xtr_seq = torch.tensor(normalize(Xtr)).unsqueeze(-1)
Xte_seq = torch.tensor(normalize(Xte)).unsqueeze(-1)


# -------------------------------------------------------------
# AUTOENCODEUR Bi-LSTM (identique au jour 6)
# -------------------------------------------------------------
class BiLSTMAutoEncodeur(nn.Module):
    def __init__(self, hidden=32, n_code=32, seq_len=L):
        super().__init__()
        self.seq_len = seq_len
        self.enc_lstm = nn.LSTM(1, hidden, batch_first=True, bidirectional=True)
        self.to_code = nn.Linear(2*hidden, n_code)
        self.from_code = nn.Linear(n_code, 2*hidden)
        self.dec_lstm = nn.LSTM(2*hidden, hidden, batch_first=True, bidirectional=True)
        self.reconstruct = nn.Linear(2*hidden, 1)
    def encode(self, x):
        out, _ = self.enc_lstm(x)
        return self.to_code(out.mean(dim=1))
    def forward(self, x):
        code = self.encode(x)
        h = self.from_code(code).unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.dec_lstm(h)
        return self.reconstruct(out), code

ae = BiLSTMAutoEncodeur()
opt_ae = torch.optim.Adam(ae.parameters(), lr=0.005)
perte_recon = nn.MSELoss()
dl = DataLoader(TensorDataset(Xtr_seq), batch_size=256, shuffle=True)

print("ETAGE 1 — autoencodeur Bi-LSTM (sur les records d'entrainement seulement)")
for epoch in range(1, 16):
    ae.train(); t0=time.time(); pertes=[]
    for (xb,) in dl:
        opt_ae.zero_grad(); recon,_ = ae(xb)
        p = perte_recon(recon, xb); p.backward(); opt_ae.step(); pertes.append(p.item())
    if epoch % 5 == 0 or epoch == 1:
        print(f"  epoque {epoch:2d} | perte {np.mean(pertes):.4f} | {time.time()-t0:.0f}s")

# IMPORTANT : le StandardScaler est ajuste SUR LE TRAIN UNIQUEMENT,
# puis applique au test. Sinon, ce serait encore une petite fuite !
ae.eval()
with torch.no_grad():
    Ctr = ae.encode(Xtr_seq).numpy()
    Cte = ae.encode(Xte_seq).numpy()
sc = StandardScaler().fit(Ctr)          # fit sur TRAIN seulement
Ctr_s = sc.transform(Ctr).astype(np.float32)
Cte_s = sc.transform(Cte).astype(np.float32)


# -------------------------------------------------------------
# CLASSIFIEUR — petit reseau
# -------------------------------------------------------------
class PetitReseau(nn.Module):
    def __init__(self, n_in=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in,32), nn.ReLU(), nn.Linear(32,5))
    def forward(self,x): return self.net(x)

clf = PetitReseau(Ctr_s.shape[1])
counts = np.bincount(ytr, minlength=5)
w = 1.0/np.sqrt(counts); w = w/w.sum()*5
loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
opt = torch.optim.Adam(clf.parameters(), lr=0.005)
dl2 = DataLoader(TensorDataset(torch.tensor(Ctr_s), torch.tensor(ytr, dtype=torch.long)),
                 batch_size=128, shuffle=True)
print("\nETAGE 3 — classification...")
for epoch in range(40):
    clf.train()
    for xb, yb in dl2:
        opt.zero_grad(); loss_fn(clf(xb), yb).backward(); opt.step()
clf.eval()
with torch.no_grad():
    pred = clf(torch.tensor(Cte_s)).argmax(1).numpy()
mf1 = f1_score(yte, pred, average='macro')


# -------------------------------------------------------------
# RESULTAT
# -------------------------------------------------------------
print("\n" + "=" * 56)
print("  macro F1 INTER-PATIENT (honnete) : %.3f" % mf1)
print("  (rappel jour 6, intra-patient, optimiste : ~0.87)")
print("=" * 56)
print("\nDetail par classe (sur des patients JAMAIS vus) :")
print(classification_report(yte, pred,
      target_names=[idx2cls[i] for i in range(5)], zero_division=0))


# =============================================================
#  CE QUE TU DOIS RETENIR
# =============================================================
# 1. Le score a sans doute baisse. C'est NORMAL et c'est SAIN : il est
#    enfin honnete, car le modele est teste sur des patients inconnus.
# 2. Deux fuites evitees ici : (a) split par record, (b) le scaler
#    ajuste sur le train seulement. Note-les dans ton journal.
# 3. Regarde quelles classes chutent le plus : souvent S et F. C'est
#    LE vrai defi clinique, et c'est ce que SMHFL cherchera a ameliorer.
#
# POUR TON JOURNAL DE DIVERGENCES (a ecrire) :
#  - "Evaluation inter-record (DS1/DS2 simplifie), pas intra-patient."
#  - "Certaines paires de records = meme patient (ex. 201/202) :
#     non traite ici, a raffiner pour une repro 100% fidele."
#  - "StandardScaler ajuste sur le train uniquement (pas de fuite)."
