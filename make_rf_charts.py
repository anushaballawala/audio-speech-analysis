import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

UD = "/userdata/msharma"
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


def chart(items, title, out):
    fig, ax = plt.subplots(1, len(items), figsize=(4.4 * len(items), 4.2))
    if len(items) == 1:
        ax = [ax]
    for a, (name, y, p) in zip(ax, items):
        a.scatter(y, p, s=16, alpha=0.5, color="#4C72B0")
        lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
        a.plot([lo, hi], [lo, hi], "r--", lw=1)
        a.set_title(f"{name}\nR2={r2_score(y, p):.2f}", fontsize=9)
        a.set_xlabel("actual"); a.set_ylabel("predicted")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


# ---- PR05 Stage2+3: stratified k-fold (4 headline targets, matches the existing grouped-by-day chart) ----
d = pd.read_csv(f"{UD}/ALL_patients_features_scores.csv")
d = d[d.patient_stage.isin(["PR05 Stage 2", "PR05 Stage 3"])].reset_index(drop=True)
SC = ["hamd_total", "hamd_q1", "hamd_q2", "hamd_q3", "hamd_q4", "hamd_q5", "hamd_q6",
      "vas_anxiety", "vas_depression", "madrs_total"]
meta = {"patient_stage", "audio_id", "timestamp"} | set(SC) | {"vas_lowenergy"}
feats = [c for c in d.columns if c not in meta]
Xa = d[feats].astype(float); Xa = Xa.fillna(Xa.median())
items = []
for s in ["hamd_total", "madrs_total", "vas_depression", "vas_anxiety"]:
    m = d[s].notna().to_numpy()
    X, y = Xa[m].to_numpy(), d[s][m].to_numpy()
    skf = StratifiedKFold(5, shuffle=True, random_state=0).split(X, strata(pd.Series(y)))
    p = cross_val_predict(RF(), X, y, cv=list(skf), n_jobs=-1)
    items.append((s, y, p))
chart(items, "PR05 Stage 2+3 — Random Forest, out-of-fold (STRATIFIED k-fold)",
      f"{UD}/pr05_rf_model/rf_pred_vs_actual_stratified.png")

# ---- RCS08: plain (non-stratified) k-fold, all 5 scores ----
P = f"{UD}/sub-RCS08-stage-1_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots"
PRE = "sub-RCS08_stage-1_audio-audiotype_preproc_spectral_gating_100_percent"
SCORES = ["pain_nrs_s1_daily", "pain_vas_s1_daily", "relief_vas_s1_daily", "mpq_s1_daily", "mood_vas_s1_weekly"]
r = pd.read_csv(f"{P}/{PRE}_pain_mood_correlation/merged_scores_features.csv")
drop = {"audio_id", "audio_uploaded", "daily_gap_days"} | set(SCORES)
fcols = [c for c in r.columns if c not in drop and r[c].dtype != object]
Xr = r[fcols].astype(float); Xr = Xr.fillna(Xr.median())
items = []
for s in SCORES:
    y = r[s].astype(float).to_numpy(); X = Xr.to_numpy()
    kf = KFold(5, shuffle=True, random_state=0).split(X, y)
    p = cross_val_predict(RF(), X, y, cv=list(kf), n_jobs=-1)
    items.append((s, y, p))
chart(items, "RCS08 (chronic pain) — Random Forest, out-of-fold (PLAIN / non-stratified k-fold)",
      f"{P}/{PRE}_rf_plots/rf_pred_vs_actual_plain_kfold.png")
