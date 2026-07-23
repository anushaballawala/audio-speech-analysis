import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(0); np.random.seed(0)   # same seed/split as rnn_pr05.py
UD = "/userdata/msharma"; L = 8
SCORES = ["hamd_total", "hamd_q1", "hamd_q2", "hamd_q3", "hamd_q4", "hamd_q5",
          "hamd_q6", "vas_anxiety", "vas_depression", "madrs_total"]
SHOW = ["hamd_total", "madrs_total", "vas_depression"]

df = pd.read_csv(f"{UD}/ALL_patients_features_scores.csv", parse_dates=["timestamp"])
df = df[df.patient_stage.isin(["PR05 Stage 2", "PR05 Stage 3"])].reset_index(drop=True)
meta = {"patient_stage", "audio_id", "timestamp"} | set(SCORES) | {"vas_lowenergy"}
feats = [c for c in df.columns if c not in meta]
df[feats] = df[feats].astype(float).fillna(df[feats].astype(float).median())


def windows(stage_df, target):
    d = stage_df.sort_values("timestamp").reset_index(drop=True)
    F = d[feats].to_numpy(float); Xs, ys, ts = [], [], []
    for i in range(L - 1, len(d)):
        if pd.isna(d[target].iloc[i]):
            continue
        Xs.append(F[i - L + 1:i + 1]); ys.append(d[target].iloc[i]); ts.append(d["timestamp"].iloc[i])
    return np.stack(Xs), np.array(ys, float), np.array(ts)


class RNN(nn.Module):  # exact architecture from rnn_pr05.py (dropout 0.2)
    def __init__(self, n_feat, hidden=64, layers=2, p=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=layers, batch_first=True, dropout=p)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(p), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def splits(target):
    Xtr_l, ytr_l, Xte_l, yte_l = [], [], [], []
    for stage in ["PR05 Stage 2", "PR05 Stage 3"]:
        X, y, t = windows(df[df.patient_stage == stage], target)
        o = np.argsort(t); X, y = X[o], y[o]; c = int(len(X) * 0.8)
        Xtr_l.append(X[:c]); ytr_l.append(y[:c]); Xte_l.append(X[c:]); yte_l.append(y[c:])
    return (np.concatenate(Xtr_l), np.concatenate(ytr_l),
            np.concatenate(Xte_l), np.concatenate(yte_l))


fig, axes = plt.subplots(len(SHOW), 2, figsize=(9, 4.2 * len(SHOW)))
for row, target in enumerate(SHOW):
    Xtr, ytr, Xte, yte = splits(target)
    sc = StandardScaler().fit(Xtr.reshape(-1, len(feats)))
    tf = lambda A: torch.tensor(sc.transform(A.reshape(-1, len(feats))).reshape(A.shape), dtype=torch.float32)
    net = RNN(len(feats))
    net.load_state_dict(torch.load(f"{UD}/pr05_rnn_model/rnn_{target}.pt"))
    net.eval()
    with torch.no_grad():
        ptr = net(tf(Xtr)).numpy(); pte = net(tf(Xte)).numpy()
    r2tr, r2te = r2_score(ytr, ptr), r2_score(yte, pte)
    print(f"{target:16s} TRAIN R2={r2tr:.3f}   TEST R2={r2te:.3f}")
    for col, (name, y, p, r2, color) in enumerate(
            [("TRAIN (data it was fit on)", ytr, ptr, r2tr, "#2A9D3F"),
             ("TEST (held-out / out-of-fold)", yte, pte, r2te, "#C44E52")]):
        ax = axes[row, col]
        ax.scatter(y, p, s=16, alpha=0.5, color=color)
        lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(f"{target} — {name}\nR2={r2:.2f}  (n={len(y)})", fontsize=10)
        ax.set_xlabel(f"actual {target}"); ax.set_ylabel("LSTM predicted")
fig.suptitle("The ACTUAL PR05 LSTM (dropout 0.2, 300 epochs — the model with the reported R2):\n"
             "predictions on its own TRAINING data vs held-out TEST", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = f"{UD}/pr05_rnn_model/rnn_actual_train_vs_test.png"
fig.savefig(out, dpi=150); plt.close(fig)
print("wrote", out)
