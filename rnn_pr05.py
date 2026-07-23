import os, numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(0); np.random.seed(0)
UD = "/userdata/msharma"; OUT = f"{UD}/pr05_rnn_model"; os.makedirs(OUT, exist_ok=True)
L = 8                     # window length (recordings of history incl. current)
SCORES = ["hamd_total", "hamd_q1", "hamd_q2", "hamd_q3", "hamd_q4", "hamd_q5",
          "hamd_q6", "vas_anxiety", "vas_depression", "madrs_total"]

df = pd.read_csv(f"{UD}/ALL_patients_features_scores.csv", parse_dates=["timestamp"])
df = df[df.patient_stage.isin(["PR05 Stage 2", "PR05 Stage 3"])].reset_index(drop=True)
meta = {"patient_stage", "audio_id", "timestamp"} | set(SCORES) | {"vas_lowenergy"}
feats = [c for c in df.columns if c not in meta]
df[feats] = df[feats].astype(float).fillna(df[feats].astype(float).median())
print(f"data: {len(df)} recordings, {len(feats)} features, window L={L}")


def windows_for(stage_df, target):
    d = stage_df.sort_values("timestamp").reset_index(drop=True)
    F = d[feats].to_numpy(float)
    Xs, ys, ts = [], [], []
    for i in range(L - 1, len(d)):
        yv = d[target].iloc[i]
        if pd.isna(yv):
            continue
        Xs.append(F[i - L + 1:i + 1]); ys.append(yv); ts.append(d["timestamp"].iloc[i])
    if not Xs:
        return None
    return np.stack(Xs), np.array(ys, float), np.array(ts)


class RNN(nn.Module):
    def __init__(self, n_feat, hidden=64, layers=2, p=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=layers, batch_first=True, dropout=p)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(p), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)   # last timestep -> score


def run_target(target):
    Xtr_l, ytr_l, Xte_l, yte_l = [], [], [], []
    for stage in ["PR05 Stage 2", "PR05 Stage 3"]:
        w = windows_for(df[df.patient_stage == stage], target)
        if w is None:
            continue
        X, y, t = w
        order = np.argsort(t); X, y = X[order], y[order]
        cut = int(len(X) * 0.8)                       # first 80% (earliest) train, last 20% test
        Xtr_l.append(X[:cut]); ytr_l.append(y[:cut]); Xte_l.append(X[cut:]); yte_l.append(y[cut:])
    Xtr, ytr = np.concatenate(Xtr_l), np.concatenate(ytr_l)
    Xte, yte = np.concatenate(Xte_l), np.concatenate(yte_l)

    sc = StandardScaler().fit(Xtr.reshape(-1, len(feats)))
    tf = lambda A: torch.tensor(sc.transform(A.reshape(-1, len(feats))).reshape(A.shape), dtype=torch.float32)
    Xtr_t, Xte_t = tf(Xtr), tf(Xte)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)

    net = RNN(len(feats))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.MSELoss()
    net.train()
    for ep in range(300):
        opt.zero_grad()
        loss = lossf(net(Xtr_t), ytr_t)
        loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(Xte_t).numpy()
    base = np.sqrt(mean_squared_error(yte, np.full_like(yte, ytr.mean())))
    torch.save(net.state_dict(), f"{OUT}/rnn_{target}.pt")
    return {"score": target, "n_train": len(ytr), "n_test": len(yte),
            "test_R2": round(r2_score(yte, pred), 3),
            "test_RMSE": round(np.sqrt(mean_squared_error(yte, pred)), 3),
            "test_MAE": round(mean_absolute_error(yte, pred), 3),
            "baseline_RMSE": round(base, 3)}, (yte, pred)


rows, store = [], {}
for s in SCORES:
    r, yp = run_target(s)
    rows.append(r); store[s] = yp
    print(f"  {s:15s} trainN={r['n_train']:3d} testN={r['n_test']:3d}  "
          f"testR2={r['test_R2']:.3f}  RMSE={r['test_RMSE']:.3f} (base {r['baseline_RMSE']:.3f})")

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/rnn_cv_metrics.csv", index=False)
print("\n=== 2-layer LSTM, held-out (last 20% by time) ===")
print(res.to_string(index=False))

show = ["hamd_total", "madrs_total", "vas_depression", "vas_anxiety"]
fig, ax = plt.subplots(1, len(show), figsize=(4.6 * len(show), 4.4))
for a, s in zip(ax, show):
    y, p = store[s]
    a.scatter(y, p, s=16, alpha=0.5, color="#8172B3")
    lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
    a.plot([lo, hi], [lo, hi], "r--", lw=1)
    a.set_title(f"{s}\ntest R²={r2_score(y, p):.2f}"); a.set_xlabel("actual"); a.set_ylabel("predicted")
fig.suptitle("PR05 Stage 2+3 — 2-layer LSTM, held-out predictions (last 20% by time)", fontsize=13)
fig.tight_layout(); fig.savefig(f"{OUT}/rnn_pred_vs_actual.png", dpi=150); plt.close(fig)
print(f"\nwrote models + rnn_cv_metrics.csv + rnn_pred_vs_actual.png to {OUT}")
