#!/usr/bin/env python3
"""
Predict symptom scores from the 41 acoustic features, per
patient x stage. Outputs CROSS-VALIDATED performance

  <parent>/<prefix>_pls_metadata/pls_results.csv        -- per target: n, best_n_components,
        cv_R2, cv_pearson_r/p (out-of-fold pred vs actual), in_sample_R2 (overfit ref)
  <parent>/<prefix>_pls_metadata/pls_params.json        -- inputs (features, CV scheme, etc.)
  <parent>/<prefix>_pls_metadata/pls_pred_<target>.csv  -- id, actual, cv_predicted
  <parent>/<prefix>_pls_metadata/pls_coef_<target>.csv  -- standardized coefficient per feature
  <parent>/<prefix>_pls_plots/pls_pred_vs_actual.png
"""

import os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score

# reuse loaders + constants from tsne_features.py (exec everything before __main__)
_ns = {}
exec(open("/userdata/msharma/audio-speech-analysis/tsne_features.py").read().split("if __name__")[0], _ns)
load_features = _ns["load_features"]; FEATURE_COLS = _ns["FEATURE_COLS"]
SUBJECT_RE = _ns["SUBJECT_RE"]; CLINICIAN_RE = _ns["CLINICIAN_RE"]; UD = _ns["UD"]
scores_from_xlsx = _ns["scores_from_xlsx"]; scores_from_csv = _ns["scores_from_csv"]
scores_from_clinician_audit = _ns["scores_from_clinician_audit"]; scores_from_pr09 = _ns["scores_from_pr09"]
S2P, S2X, S3P, S3X, CLP, CLX, CL_AUDIT, XLSX = (_ns[k] for k in ["S2P","S2X","S3P","S3X","CLP","CLX","CL_AUDIT","XLSX"])

TARGETS = ["hamd_total", "madrs_total", "vas_anxiety", "vas_depression"]
MAX_COMP = 10


def pls_for_target(X, y):
    n = len(y)
    n_splits = 5 if n >= 25 else max(3, min(5, n // 3))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    best = None
    for nc in range(1, min(MAX_COMP, X.shape[1], n - n_splits) + 1):
        pipe = make_pipeline(StandardScaler(), PLSRegression(n_components=nc))
        yhat = cross_val_predict(pipe, X, y, cv=cv).ravel()
        r2 = r2_score(y, yhat)
        r, p = pearsonr(y, yhat)
        if best is None or r2 > best["cv_r2"]:
            best = dict(n_components=nc, cv_r2=r2, cv_r=r, cv_p=p, yhat=yhat, n_splits=n_splits)
    # in-sample (overfit reference) + coefficients from full-data fit at best nc
    pipe = make_pipeline(StandardScaler(), PLSRegression(n_components=best["n_components"]))
    pipe.fit(X, y)
    ins_r2 = r2_score(y, pipe.predict(X).ravel())
    coef = pipe.named_steps["plsregression"].coef_.ravel()  # standardized-X coefficients
    return best, ins_r2, coef


def run(name, parent, prefix, id_re, score_df, out_parent, out_prefix):
    feats = load_features(parent, prefix, id_re)
    feats["id"] = feats["id"].astype(str)
    s = score_df.copy(); s["id"] = s["id"].astype(str)
    df = feats.merge(s, on="id", how="inner")
    Xall = df.reindex(columns=FEATURE_COLS).apply(pd.to_numeric, errors="coerce")

    md = f"{out_parent}/{out_prefix}_pls_metadata"; pl = f"{out_parent}/{out_prefix}_pls_plots"
    os.makedirs(md, exist_ok=True); os.makedirs(pl, exist_ok=True)

    results = []
    targets = [t for t in TARGETS if t in df.columns and pd.to_numeric(df[t], errors="coerce").notna().sum() >= 15]
    fig, axes = plt.subplots(1, max(1, len(targets)), figsize=(4.6 * max(1, len(targets)), 4.4), squeeze=False)
    for ax, t in zip(axes[0], targets):
        y = pd.to_numeric(df[t], errors="coerce")
        mask = y.notna() & Xall.notna().all(axis=1)   # drop silent recordings (NaN features) + missing target
        X = Xall[mask].to_numpy(float); yv = y[mask].to_numpy(float)
        best, ins_r2, coef = pls_for_target(X, yv)
        results.append(dict(target=t, n=int(mask.sum()), best_n_components=best["n_components"],
                            cv_r2=round(best["cv_r2"], 4), cv_pearson_r=round(best["cv_r"], 4),
                            cv_pearson_p=best["cv_p"], in_sample_r2=round(ins_r2, 4), n_splits=best["n_splits"]))
        pd.DataFrame({"id": df["id"][mask].values, "actual": yv, "cv_predicted": best["yhat"]}).to_csv(
            f"{md}/pls_pred_{t}.csv", index=False)
        pd.DataFrame({"feature": FEATURE_COLS, "std_coef": coef}).reindex(
            pd.Series(np.abs(coef)).sort_values(ascending=False).index).to_csv(f"{md}/pls_coef_{t}.csv", index=False)
        ax.scatter(yv, best["yhat"], s=18, alpha=0.6)
        lo, hi = min(yv.min(), best["yhat"].min()), max(yv.max(), best["yhat"].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel(f"actual {t}"); ax.set_ylabel("CV-predicted")
        ax.set_title(f"{t}\nCV R2={best['cv_r2']:.2f} r={best['cv_r']:.2f} n={int(mask.sum())} (nc={best['n_components']})", fontsize=9)
    fig.suptitle(f"{name}: PLS regression (acoustic features -> symptoms), out-of-fold", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{pl}/pls_pred_vs_actual.png", dpi=150); plt.close(fig)

    res = pd.DataFrame(results)
    res.to_csv(f"{md}/pls_results.csv", index=False)
    json.dump(dict(n_features=len(FEATURE_COLS), feature_columns=FEATURE_COLS,
                   model="sklearn.cross_decomposition.PLSRegression",
                   preprocessing="StandardScaler inside CV pipeline (no leakage); rows with any NaN feature or missing target dropped",
                   cv="KFold(shuffle, random_state=0)", n_components_scanned=f"1..{MAX_COMP}",
                   component_selection="max out-of-fold R2", targets=targets),
              open(f"{md}/pls_params.json", "w"), indent=2, default=str)
    print(f"\n### {name} ###")
    print(res.to_string(index=False))
    return res


if __name__ == "__main__":
    P8 = f"{UD}/sub-PR08-stage-2_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots"
    X8 = "sub-PR08_stage-2_audio-audiotype_preproc_spectral_gating_100_percent"
    P9 = f"{UD}/sub-PR09-stage-2_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots"
    X9 = "sub-PR09_stage-2_audio-audiotype_preproc_spectral_gating_100_percent"
    INDEX9 = "/data_store2/resection/neuropsych_video/presidio/Stage2/PR09/home/Files_PR09Stage2_2026-04-18_1541/index.html"

    run("PR05 Stage 2", S2P, S2X, SUBJECT_RE,
        scores_from_xlsx(XLSX, "Stage 2 AudioScore Match", "audio_task_timestamp"), S2P, S2X)
    run("PR05 Stage 3", S3P, S3X, SUBJECT_RE,
        scores_from_csv(f"{UD}/PR05Stage3_DATA_2026-06-07_1941.csv",
                        ["audio_task_timestamp", "start_local_timestamp", "completion_pt_timestamp"]), S3P, S3X)
    run("PR05 Clinician", CLP, CLX, CLINICIAN_RE, scores_from_clinician_audit(CL_AUDIT), CLP, CLX)
    run("PR08 Stage 2", P8, X8, SUBJECT_RE,
        scores_from_csv(f"{UD}/PR08PreStage2_DATA_2026-04-18_1616.csv",
                        ["audio_task_timestamp", "start_timestamp_local", "completion_pt_timestamp"]), P8, X8)
    run("PR09 Stage 2", P9, X9, SUBJECT_RE, scores_from_pr09(f"{UD}/PR09Stage2_DATA_2026-04-18_1544.csv", INDEX9), P9, X9)
