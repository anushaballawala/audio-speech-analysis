import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
torch.manual_seed(0); np.random.seed(0)
UD = "/userdata/msharma"; L = 8
SCORES = ["hamd_total", "hamd_q1", "hamd_q2", "hamd_q3", "hamd_q4", "hamd_q5",
          "hamd_q6", "vas_anxiety", "vas_depression", "madrs_total"]
df = pd.read_csv(f"{UD}/ALL_patients_features_scores.csv", parse_dates=["timestamp"])
df = df[df.patient_stage.isin(["PR05 Stage 2", "PR05 Stage 3"])].reset_index(drop=True)
meta = {"patient_stage", "audio_id", "timestamp"} | set(SCORES) | {"vas_lowenergy"}
feats = [c for c in df.columns if c not in meta]
df[feats] = df[feats].astype(float).fillna(df[feats].astype(float).median())


def windows(sd, t):
    d = sd.sort_values("timestamp").reset_index(drop=True); F = d[feats].to_numpy(float); Xs, ys, ts = [], [], []
    for i in range(L - 1, len(d)):
        if pd.isna(d[t].iloc[i]):
            continue
        Xs.append(F[i - L + 1:i + 1]); ys.append(d[t].iloc[i]); ts.append(d["timestamp"].iloc[i])
    return np.stack(Xs), np.array(ys, float), np.array(ts)


class RNN(nn.Module):  # converged model shown in rnn_train_vs_test_overfit.png (dropout off)
    def __init__(s, nf, h=64, l=2, p=0.0):
        super().__init__(); s.lstm = nn.LSTM(nf, h, l, batch_first=True, dropout=p)
        s.head = nn.Sequential(nn.Linear(h, 32), nn.ReLU(), nn.Dropout(p), nn.Linear(32, 1))

    def forward(s, x):
        o, _ = s.lstm(x); return s.head(o[:, -1, :]).squeeze(-1)


def sp(t):
    a, b, c, d2 = [], [], [], []
    for st in ["PR05 Stage 2", "PR05 Stage 3"]:
        X, y, ts = windows(df[df.patient_stage == st], t); o = np.argsort(ts); X, y = X[o], y[o]; cut = int(len(X) * 0.8)
        a.append(X[:cut]); b.append(y[:cut]); c.append(X[cut:]); d2.append(y[cut:])
    return np.concatenate(a), np.concatenate(b), np.concatenate(c), np.concatenate(d2)


hdr = ["target", "pearson_r", "r^2", "R2(det)", "RMSE", "baseRMSE=std(act)", "std(pred)", "mean(act)", "mean(pred)"]
print("  ".join(f"{h:>12}" for h in hdr))
for t in ["hamd_total", "madrs_total", "vas_depression"]:
    Xtr, ytr, Xte, yte = sp(t)
    sc = StandardScaler().fit(Xtr.reshape(-1, len(feats)))
    tf = lambda A: torch.tensor(sc.transform(A.reshape(-1, len(feats))).reshape(A.shape), dtype=torch.float32)
    net = RNN(len(feats))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4); lossf = nn.MSELoss()
    ytr_t = torch.tensor(ytr, dtype=torch.float32); net.train()
    for _ in range(1500):
        opt.zero_grad(); lossf(net(tf(Xtr)), ytr_t).backward(); opt.step()
    net.eval()
    with torch.no_grad():
        p = net(tf(Xte)).numpy()
    r, _ = pearsonr(yte, p); R2 = r2_score(yte, p); rmse = np.sqrt(mean_squared_error(yte, p)); base = yte.std()
    vals = [t, f"{r:.3f}", f"{r**2:.3f}", f"{R2:.3f}", f"{rmse:.2f}", f"{base:.2f}", f"{p.std():.2f}", f"{yte.mean():.2f}", f"{p.mean():.2f}"]
    print("  ".join(f"{v:>12}" for v in vals))
