import os
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

UD = "/userdata/msharma"
OUT = f"{UD}/pr05_rf_model"
os.makedirs(OUT, exist_ok=True)
SCORES = ["hamd_total", "hamd_q1", "hamd_q2", "hamd_q3", "hamd_q4", "hamd_q5",
          "hamd_q6", "vas_anxiety", "vas_depression", "madrs_total"]

df = pd.read_csv(f"{UD}/ALL_patients_features_scores.csv", parse_dates=["timestamp"])
df = df[df.patient_stage.isin(["PR05 Stage 2", "PR05 Stage 3"])].reset_index(drop=True)
meta = {"patient_stage", "audio_id", "timestamp"} | set(SCORES) | {"vas_lowenergy"}
feats = [c for c in df.columns if c not in meta]
print(f"data: {len(df)} recordings, {len(feats)} features  "
      f"({(df.patient_stage=='PR05 Stage 2').sum()} S2 + {(df.patient_stage=='PR05 Stage 3').sum()} S3)")

X_all = df[feats].astype(float)
X_all = X_all.fillna(X_all.median())          # impute the handful of NaN feature cells
day = df.timestamp.dt.floor("D").astype("int64").to_numpy()   # group key = calendar day

RF = lambda: RandomForestRegressor(n_estimators=600, min_samples_leaf=2,
                                   max_features="sqrt", n_jobs=-1, random_state=0)

rows, oof_store = [], {}
for s in SCORES:
    m = df[s].notna().to_numpy()
    if m.sum() < 30:
        print(f"  skip {s}: only {m.sum()} labelled")
        continue
    X, y, g = X_all[m].to_numpy(), df[s][m].to_numpy(), day[m]
    base_rmse = np.sqrt(mean_squared_error(y, np.full_like(y, y.mean())))
    for tag, cv in [("kfold", KFold(5, shuffle=True, random_state=0)),
                    ("group_day", GroupKFold(min(5, len(np.unique(g)))))]:
        pred = cross_val_predict(RF(), X, y, cv=cv, groups=g if tag == "group_day" else None, n_jobs=-1)
        rows.append({"score": s, "cv": tag, "n": int(m.sum()),
                     "R2": round(r2_score(y, pred), 3),
                     "RMSE": round(np.sqrt(mean_squared_error(y, pred)), 3),
                     "MAE": round(mean_absolute_error(y, pred), 3),
                     "baseline_RMSE": round(base_rmse, 3)})
        if tag == "group_day":
            oof_store[s] = (y, pred)

res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/rf_cv_metrics.csv", index=False)
print("\n=== out-of-fold performance ===")
print(res.to_string(index=False))

# final models on all data + feature importances (avg across targets)
imp = pd.DataFrame(index=feats)
for s in SCORES:
    m = df[s].notna().to_numpy()
    if m.sum() < 30:
        continue
    rf = RF().fit(X_all[m], df[s][m])
    joblib.dump(rf, f"{OUT}/rf_{s}.joblib")
    imp[s] = rf.feature_importances_
imp["mean"] = imp.mean(axis=1)
imp = imp.sort_values("mean", ascending=False)
imp.to_csv(f"{OUT}/rf_feature_importance.csv")
print("\ntop 15 features by mean importance across targets:")
print(imp["mean"].head(15).round(4).to_string())

# plots: OOF (grouped-by-day) predicted vs actual for the headline targets
show = [s for s in ["hamd_total", "madrs_total", "vas_depression", "vas_anxiety"] if s in oof_store]
fig, ax = plt.subplots(1, len(show), figsize=(4.6*len(show), 4.4))
if len(show) == 1: ax = [ax]
for a, s in zip(ax, show):
    y, p = oof_store[s]
    a.scatter(y, p, s=14, alpha=0.4, color="#4C72B0")
    lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
    a.plot([lo, hi], [lo, hi], "r--", lw=1)
    r2 = r2_score(y, p)
    a.set_title(f"{s}\nOOF R²={r2:.2f} (grouped by day)")
    a.set_xlabel("actual"); a.set_ylabel("predicted")
fig.suptitle("PR05 Stage 2+3 — Random Forest, out-of-fold predictions", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/rf_pred_vs_actual.png", dpi=150); plt.close(fig)

# importance bar for hamd_total
fig, a = plt.subplots(figsize=(8, 6))
top = imp["hamd_total"].sort_values().tail(15)
a.barh(range(len(top)), top.to_numpy(), color="#55A868")
a.set_yticks(range(len(top)), top.index, fontsize=8)
a.set_title("PR05 Stage 2+3 RF — top 15 features for HAM-D total")
a.set_xlabel("Gini importance")
fig.tight_layout(); fig.savefig(f"{OUT}/rf_importance_hamd.png", dpi=150); plt.close(fig)

print(f"\nwrote models + rf_cv_metrics.csv, rf_feature_importance.csv, "
      f"rf_pred_vs_actual.png, rf_importance_hamd.png to {OUT}")
