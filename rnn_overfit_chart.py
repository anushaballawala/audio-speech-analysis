import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(0); np.random.seed(0)
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


class RNN(nn.Module):  # same architecture as rnn_pr05.py; dropout off + trained to convergence to isolate overfitting
    def __init__(self, n_feat, hidden=64, layers=2, p=0.0):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=layers, batch_first=True, dropout=p)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(p), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def fit_target(target):
    Xtr_l, ytr_l, Xte_l, yte_l = [], [], [], []
    for stage in ["PR05 Stage 2", "PR05 Stage 3"]:
        X, y, t = windows(df[df.patient_stage == stage], target)
        o = np.argsort(t); X, y = X[o], y[o]; c = int(len(X) * 0.8)
        Xtr_l.append(X[:c]); ytr_l.append(y[:c]); Xte_l.append(X[c:]); yte_l.append(y[c:])
    Xtr, ytr = np.concatenate(Xtr_l), np.concatenate(ytr_l)
    Xte, yte = np.concatenate(Xte_l), np.concatenate(yte_l)
    sc = StandardScaler().fit(Xtr.reshape(-1, len(feats)))
    tf = lambda A: torch.tensor(sc.transform(A.reshape(-1, len(feats))).reshape(A.shape), dtype=torch.float32)
    Xtr_t, Xte_t, ytr_t = tf(Xtr), tf(Xte), torch.tensor(ytr, dtype=torch.float32)
    net = RNN(len(feats))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.MSELoss(); net.train()
    for _ in range(1500):
        opt.zero_grad(); lossf(net(Xtr_t), ytr_t).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return (ytr, net(Xtr_t).numpy()), (yte, net(Xte_t).numpy())


fig, axes = plt.subplots(len(SHOW), 2, figsize=(9, 4.2 * len(SHOW)))
for row, target in enumerate(SHOW):
    (ytr, ptr), (yte, pte) = fit_target(target)
    for col, (name, y, p, color) in enumerate([("TRAIN (what it fit)", ytr, ptr, "#2A9D3F"),
                                               ("TEST (out-of-fold)", yte, pte, "#C44E52")]):
        ax = axes[row, col]
        ax.scatter(y, p, s=16, alpha=0.5, color=color)
        lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(f"{target} — {name}\nR2={r2_score(y, p):.2f}  (n={len(y)})", fontsize=10)
        ax.set_xlabel(f"actual {target}"); ax.set_ylabel("LSTM predicted")
fig.suptitle("PR05 2-layer LSTM: memorizes TRAIN (R2 high, on diagonal) but fails on held-out TEST (R2<0)\n"
             "= overfitting, not generalization", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = f"{UD}/pr05_rnn_model/rnn_train_vs_test_overfit.png"
fig.savefig(out, dpi=150); plt.close(fig)
print("wrote", out)
for t in SHOW:
    (ytr, ptr), (yte, pte) = fit_target(t)
    print(f"{t:16s} train R2={r2_score(ytr,ptr):.3f}   test R2={r2_score(yte,pte):.3f}")
