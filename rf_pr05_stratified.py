import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

UD = "/userdata/msharma"
SCORES = ["hamd_total", "hamd_q1", "hamd_q2", "hamd_q3", "hamd_q4", "hamd_q5",
          "hamd_q6", "vas_anxiety", "vas_depression", "madrs_total"]
df = pd.read_csv(f"{UD}/ALL_patients_features_scores.csv")
df = df[df.patient_stage.isin(["PR05 Stage 2", "PR05 Stage 3"])].reset_index(drop=True)
meta = {"patient_stage", "audio_id", "timestamp"} | set(SCORES) | {"vas_lowenergy"}
feats = [c for c in df.columns if c not in meta]
X_all = df[feats].astype(float)
X_all = X_all.fillna(X_all.median())
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


rows = []
for s in SCORES:
    m = df[s].notna().to_numpy()
    if m.sum() < 30:
        continue
    X, y = X_all[m].to_numpy(), df[s][m].to_numpy()
    base = np.sqrt(mean_squared_error(y, np.full_like(y, y.mean())))
    strat = strata(pd.Series(y))
    for tag, splits in [("plain_kfold", KFold(5, shuffle=True, random_state=0).split(X, y)),
                        ("stratified_kfold", StratifiedKFold(5, shuffle=True, random_state=0).split(X, strat))]:
        pred = cross_val_predict(RF(), X, y, cv=list(splits), n_jobs=-1)
        rows.append({"score": s, "cv": tag, "n": int(m.sum()), "n_strata_bins": int(pd.Series(strat).nunique()),
                     "R2": round(r2_score(y, pred), 3),
                     "RMSE": round(np.sqrt(mean_squared_error(y, pred)), 3),
                     "MAE": round(mean_absolute_error(y, pred), 3),
                     "baseline_RMSE": round(base, 3)})

res = pd.DataFrame(rows)
res.to_csv(f"{UD}/pr05_rf_model/rf_stratified_metrics.csv", index=False)
# side-by-side view
piv = res.pivot_table(index=["score", "n"], columns="cv", values="R2")
piv["delta(strat-plain)"] = (piv["stratified_kfold"] - piv["plain_kfold"]).round(3)
print("=== out-of-fold R2: plain KFold vs Stratified KFold (PR05 Stage2+3 RF) ===")
print(piv.reset_index().to_string(index=False))
print("\nfull metrics:")
print(res.to_string(index=False))
