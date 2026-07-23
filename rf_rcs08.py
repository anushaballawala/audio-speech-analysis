import os
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

UD = "/userdata/msharma"
PARENT = f"{UD}/sub-RCS08-stage-1_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots"
PREFIX = "sub-RCS08_stage-1_audio-audiotype_preproc_spectral_gating_100_percent"
SCORES = ["pain_nrs_s1_daily", "pain_vas_s1_daily", "relief_vas_s1_daily", "mpq_s1_daily", "mood_vas_s1_weekly"]
OUT_MD = f"{PARENT}/{PREFIX}_rf_metadata"; OUT_PL = f"{PARENT}/{PREFIX}_rf_plots"
os.makedirs(OUT_MD, exist_ok=True); os.makedirs(OUT_PL, exist_ok=True)

df = pd.read_csv(f"{PARENT}/{PREFIX}_pain_mood_correlation/merged_scores_features.csv")
drop = {"audio_id", "audio_uploaded", "daily_gap_days"} | set(SCORES)
feats = [c for c in df.columns if c not in drop and df[c].dtype != object]
X_all = df[feats].astype(float); X_all = X_all.fillna(X_all.median())
print(f"n={len(df)} recordings, {len(feats)} features")

RF = lambda: RandomForestRegressor(n_estimators=600, min_samples_leaf=2,
                                   max_features="sqrt", n_jobs=-1, random_state=0)


def strata(y, k=5):
    for q in range(k, 1, -1):
        try:
            b = pd.qcut(y, q=q, duplicates="drop", labels=False)
            if b.nunique() >= 2 and b.value_counts().min() >= 5:
                return b.to_numpy()
        except Exception:
            pass
    return pd.Series(np.round(y)).astype("category").cat.codes.to_numpy()


rows, oof = [], {}
for s in SCORES:
    y = df[s].astype(float).to_numpy(); X = X_all.to_numpy()
    base = np.sqrt(mean_squared_error(y, np.full_like(y, y.mean())))
    st = strata(pd.Series(y))
    for tag, splits in [("plain_kfold", KFold(5, shuffle=True, random_state=0).split(X, y)),
                        ("stratified_kfold", StratifiedKFold(5, shuffle=True, random_state=0).split(X, st))]:
        pred = cross_val_predict(RF(), X, y, cv=list(splits), n_jobs=-1)
        rows.append({"score": s, "cv": tag, "n": len(y),
                     "R2": round(r2_score(y, pred), 3),
                     "RMSE": round(np.sqrt(mean_squared_error(y, pred)), 3),
                     "MAE": round(mean_absolute_error(y, pred), 3),
                     "baseline_RMSE": round(base, 3)})
        if tag == "stratified_kfold":
            oof[s] = (y, pred)

res = pd.DataFrame(rows)
res.to_csv(f"{OUT_MD}/rf_cv_metrics.csv", index=False)
piv = res.pivot_table(index=["score", "n"], columns="cv", values="R2")
piv["delta(strat-plain)"] = (piv["stratified_kfold"] - piv["plain_kfold"]).round(3)
print("\n=== RCS08 RF out-of-fold R2: plain vs stratified KFold ===")
print(piv.reset_index().to_string(index=False))
print("\nfull metrics:")
print(res.to_string(index=False))

# feature importance (full-fit) for the two best targets + plot
imp = pd.DataFrame(index=feats)
for s in SCORES:
    imp[s] = RF().fit(X_all, df[s].astype(float)).feature_importances_
imp["mean"] = imp.mean(axis=1)
imp.sort_values("mean", ascending=False).to_csv(f"{OUT_MD}/rf_feature_importance.csv")
print("\ntop 10 features (mean importance):")
print(imp["mean"].sort_values(ascending=False).head(10).round(4).to_string())

fig, ax = plt.subplots(1, len(SCORES), figsize=(4.4 * len(SCORES), 4.2))
for a, s in zip(ax, SCORES):
    y, p = oof[s]
    a.scatter(y, p, s=16, alpha=0.5, color="#4C72B0")
    lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
    a.plot([lo, hi], [lo, hi], "r--", lw=1)
    a.set_title(f"{s}\nstratified R2={r2_score(y,p):.2f}", fontsize=9)
    a.set_xlabel("actual"); a.set_ylabel("predicted")
fig.suptitle("RCS08 (chronic pain) — Random Forest, out-of-fold (stratified KFold)", fontsize=12)
fig.tight_layout(); fig.savefig(f"{OUT_PL}/rf_pred_vs_actual.png", dpi=150); plt.close(fig)
print(f"\nwrote -> {OUT_MD} + {OUT_PL}")
