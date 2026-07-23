import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score
UD = "/userdata/msharma"
df = pd.read_csv(f"{UD}/ALL_patients_features_scores.csv", parse_dates=["timestamp"])
df = df[df.patient_stage.isin(["PR05 Stage 2", "PR05 Stage 3"])].reset_index(drop=True)
SC = ["hamd_total", "hamd_q1", "hamd_q2", "hamd_q3", "hamd_q4", "hamd_q5",
      "hamd_q6", "vas_anxiety", "vas_depression", "madrs_total"]
meta = {"patient_stage", "audio_id", "timestamp"} | set(SC) | {"vas_lowenergy"}
feats = [c for c in df.columns if c not in meta]
X = df[feats].astype(float); X = X.fillna(X.median())
day = df.timestamp.dt.floor("D").astype("int64").to_numpy()
RF = lambda: RandomForestRegressor(n_estimators=600, min_samples_leaf=2,
                                   max_features="sqrt", n_jobs=-1, random_state=0)
tol = {"hamd_total": 3, "hamd_q1": 1, "hamd_q2": 1, "hamd_q3": 1, "hamd_q4": 1,
       "hamd_q5": 1, "hamd_q6": 1, "vas_anxiety": 10, "vas_depression": 10, "madrs_total": 5}
scale = {"hamd_total": 20, "hamd_q1": 4, "hamd_q2": 4, "hamd_q3": 3, "hamd_q4": 4,
         "hamd_q5": 3, "hamd_q6": 2, "vas_anxiety": 100, "vas_depression": 100, "madrs_total": 60}
print(f"{'score':15s} {'n':>4} {'±tol':>5} {'within_tol%':>11} {'exact_int%':>11} {'R2(var%)':>9}")
for s in SC:
    m = df[s].notna().to_numpy()
    if m.sum() < 30:
        continue
    Xs, y, g = X[m].to_numpy(), df[s][m].to_numpy(), day[m]
    p = cross_val_predict(RF(), Xs, y, cv=GroupKFold(5), groups=g, n_jobs=-1)
    within = 100 * np.mean(np.abs(p - y) <= tol[s])
    exact = 100 * np.mean(np.round(p) == np.round(y))
    print(f"{s:15s} {m.sum():4d} {tol[s]:5d} {within:10.1f}% {exact:10.1f}% {100*r2_score(y,p):8.1f}%")
