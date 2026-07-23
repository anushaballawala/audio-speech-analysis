import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

torch.manual_seed(0); np.random.seed(0)
UD = "/userdata/msharma"; L = 8; TARGET = "hamd_total"
SCORES = ["hamd_total", "hamd_q1", "hamd_q2", "hamd_q3", "hamd_q4", "hamd_q5",
          "hamd_q6", "vas_anxiety", "vas_depression", "madrs_total"]
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


Xtr_l, ytr_l, Xte_l, yte_l = [], [], [], []
for stage in ["PR05 Stage 2", "PR05 Stage 3"]:
    X, y, t = windows(df[df.patient_stage == stage], TARGET)
    o = np.argsort(t); X, y = X[o], y[o]; c = int(len(X) * 0.8)
    Xtr_l.append(X[:c]); ytr_l.append(y[:c]); Xte_l.append(X[c:]); yte_l.append(y[c:])
Xtr, ytr = np.concatenate(Xtr_l), np.concatenate(ytr_l)
Xte, yte = np.concatenate(Xte_l), np.concatenate(yte_l)
sc = StandardScaler().fit(Xtr.reshape(-1, len(feats)))
tf = lambda A: torch.tensor(sc.transform(A.reshape(-1, len(feats))).reshape(A.shape), dtype=torch.float32)
Xtr_t, Xte_t = tf(Xtr), tf(Xte)
ytr_t = torch.tensor(ytr, dtype=torch.float32)
print(f"train {len(ytr)} windows, test {len(yte)} windows, feat dim {len(feats)}")


def make(kind):
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            if kind == "lstm":
                self.rec = nn.LSTM(len(feats), 64, 2, batch_first=True, dropout=0.0)
            else:
                self.rec = nn.RNN(len(feats), 64, 2, batch_first=True, nonlinearity="tanh", dropout=0.0)
            self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):
            o, _ = self.rec(x); return self.head(o[:, -1, :]).squeeze(-1)
    return Net()


def train(kind, epochs, batch, lr):
    net = make(kind)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.MSELoss()
    n = len(ytr_t)
    print(f"\n=== {kind.upper()}  epochs={epochs} batch={batch} lr={lr}  "
          f"(no dropout, so it SHOULD be able to overfit) ===")
    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(n) if batch else None
        if batch:
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                opt.zero_grad(); loss = lossf(net(Xtr_t[idx]), ytr_t[idx])
                loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        else:
            opt.zero_grad(); loss = lossf(net(Xtr_t), ytr_t)
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        if ep % max(1, epochs // 8) == 0 or ep == 1:
            net.eval()
            with torch.no_grad():
                rtr = r2_score(ytr, net(Xtr_t).numpy()); rte = r2_score(yte, net(Xte_t).numpy())
            print(f"  epoch {ep:5d}   train R2 = {rtr:7.3f}   test R2 = {rte:7.3f}")


# original recipe (full-batch, 300 epochs) vs longer vs mini-batch
train("lstm", 300, None, 1e-3)          # what we ran before
train("lstm", 4000, None, 1e-3)         # same but way more epochs
train("lstm", 800, 32, 1e-3)            # mini-batch SGD
train("rnn", 4000, None, 1e-3)
train("rnn", 800, 32, 1e-3)
