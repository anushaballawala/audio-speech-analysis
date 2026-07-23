import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
torch.manual_seed(0); np.random.seed(0)
UD = "/userdata/msharma"; L = 8; T = "hamd_total"
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


class RNN(nn.Module):
    def __init__(s, nf, h=64, l=2, p=0.0):
        super().__init__(); s.lstm = nn.LSTM(nf, h, l, batch_first=True, dropout=p)
        s.head = nn.Sequential(nn.Linear(h, 32), nn.ReLU(), nn.Dropout(p), nn.Linear(32, 1))

    def forward(s, x):
        o, _ = s.lstm(x); return s.head(o[:, -1, :]).squeeze(-1)


a, b, c, d2 = [], [], [], []
for st in ["PR05 Stage 2", "PR05 Stage 3"]:
    X, y, ts = windows(df[df.patient_stage == st], T); o = np.argsort(ts); X, y = X[o], y[o]; cut = int(len(X) * 0.8)
    a.append(X[:cut]); b.append(y[:cut]); c.append(X[cut:]); d2.append(y[cut:])
Xtr, ytr, Xte, yte = np.concatenate(a), np.concatenate(b), np.concatenate(c), np.concatenate(d2)
sc = StandardScaler().fit(Xtr.reshape(-1, len(feats)))
tf = lambda A: torch.tensor(sc.transform(A.reshape(-1, len(feats))).reshape(A.shape), dtype=torch.float32)
net = RNN(len(feats)); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4); lf = nn.MSELoss()
yt = torch.tensor(ytr, dtype=torch.float32); net.train()
for _ in range(1500):
    opt.zero_grad(); lf(net(tf(Xtr)), yt).backward(); opt.step()
net.eval()
with torch.no_grad():
    p = net(tf(Xte)).numpy()

ybar = yte.mean()
err_diag = p - yte                       # vertical miss from y=x  (model error)
err_mean = ybar - yte                    # error of predict-the-mean
slope, intercept = np.polyfit(yte, p, 1)
rmse_d = np.sqrt(np.mean(err_diag**2)); rmse_m = np.sqrt(np.mean(err_mean**2))
r, _ = pearsonr(yte, p)
print(f"n_test={len(yte)}  Pearson r={r:.3f}  R2={r2_score(yte,p):.3f}")
print(f"RMSE to diagonal (model error) = {rmse_d:.2f}")
print(f"RMSE of predict-the-mean       = {rmse_m:.2f}   (=std of actual)")
print(f"best-fit line of pred~actual: slope={slope:.2f}, intercept={intercept:.2f}  (diagonal would be slope=1,int=0)")
print(f"median |pred-actual| = {np.median(np.abs(err_diag)):.2f} points")
print(f"share of test points within 2 pts of y=x: {np.mean(np.abs(err_diag)<=2)*100:.0f}%")
print(f"share of points CLOSER to the mean line than to y=x: {np.mean(np.abs(err_mean)<np.abs(err_diag))*100:.0f}%")

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
# left: scatter with the three lines
lo, hi = min(yte.min(), p.min()) - 1, max(yte.max(), p.max()) + 1
ax[0].scatter(yte, p, s=26, alpha=0.5, color="#C44E52", zorder=3)
xs = np.array([lo, hi])
ax[0].plot(xs, xs, "k--", lw=1.6, label="y = x  (perfect prediction)")
ax[0].axhline(ybar, color="#4C72B0", lw=1.6, label=f"predict-the-mean  (ŷ={ybar:.1f}, RMSE {rmse_m:.1f})")
ax[0].plot(xs, slope * xs + intercept, color="#2A9D3F", lw=1.6, label=f"best fit of the cloud (slope {slope:.2f})")
ax[0].set_xlim(lo, hi); ax[0].set_ylim(lo, hi); ax[0].set_aspect("equal")
ax[0].set_xlabel("actual hamd_total"); ax[0].set_ylabel("LSTM predicted")
ax[0].set_title(f"hamd_total TEST: the cloud's best fit (slope {slope:.2f}) is far flatter than y=x\n"
                f"Pearson r={r:.2f} but R²={r2_score(yte,p):.2f}")
ax[0].legend(fontsize=8, loc="upper left")
# vertical residual segments to the diagonal (the errors R² squares)
for xi, yi in zip(yte, p):
    ax[0].plot([xi, xi], [xi, yi], color="gray", lw=0.5, alpha=0.5, zorder=1)

# right: |error| comparison
ax[1].hist(np.abs(err_diag), bins=15, alpha=0.6, color="#C44E52", label=f"|error| of LSTM (RMSE {rmse_d:.1f})")
ax[1].axvline(rmse_d, color="#C44E52", ls="--"); ax[1].axvline(rmse_m, color="#4C72B0", ls="--")
ax[1].hist(np.abs(err_mean), bins=15, alpha=0.5, color="#4C72B0", label=f"|error| of predict-the-mean (RMSE {rmse_m:.1f})")
ax[1].set_xlabel("absolute error (HAM-D points)"); ax[1].set_ylabel("# test recordings")
ax[1].set_title("The LSTM's errors are LARGER than just guessing the mean\n(dashed = RMSE of each) -> that is why R² < 0")
ax[1].legend(fontsize=9)
fig.tight_layout()
out = f"{UD}/pr05_rnn_model/rnn_diagonal_vs_mean_hamd.png"
fig.savefig(out, dpi=150); plt.close(fig)
print("wrote", out)
