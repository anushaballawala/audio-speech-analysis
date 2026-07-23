#!/usr/bin/env python3
"""t-SNE of all acoustic features per patient, colored by time / MADRS / VAS-A / VAS-D.

For each patient the full feature vector (all 11 families = 41 columns) of every
recording is standardized and embedded into 2-D with t-SNE.


  <parent>/<prefix>_tsne_metadata/tsne_embedding.csv  -- per-recording id, stage,
        timestamp, day_number, tsne_x, tsne_y, and the score columns used.
  <parent>/<prefix>_tsne_metadata/tsne_params.json    -- all t-SNE inputs + the
        feature list / preprocessing (the "inputs to the functions" record).
  <parent>/<prefix>_tsne_plots/tsne_4panel.png
"""

import re, glob, os, json, html as html_lib
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

UD = "/userdata/msharma"
SUBJECT_RE = re.compile(r"signal-preproc_(?:wiener_)?(\d+)_")
CLINICIAN_RE = re.compile(r"(GMT\d{8}-\d{6}_Recording)")

FAMILY_SPEC = {
    "pitch": ("_pitches.csv", ["pitch_mean", "pitch_std", "pitch_median", "pitch_iqr"]),
    "loudness": ("_loudness_in_db.csv", ["active_intensity_vals_mean", "intensity_std", "intensity_median", "intensity_iqr"]),
    "f3": ("_relative_energy_formant.csv", ["mean_rel_energy_f_i", "rel_energy_std", "rel_energy_median", "rel_energy_iqr"]),
    "alpha_ratio": ("_alpha_ratio.csv", ["alpha_ratio_mean", "alpha_ratio_std", "alpha_ratio_median", "alpha_ratio_iqr"]),
    "jitter": ("_jitter.csv", ["jitter_val"]),
    "shimmer": ("_shimmer_apqN.csv", ["shimmer_val"]),
    "wpm": ("_wpm.csv", ["wpm"]),
    "hnr": ("_hnr.csv", ["hnr_mean", "hnr_std", "hnr_median", "hnr_iqr"]),
    "zcr": ("_zcr.csv", ["zcr_mean", "zcr_std", "zcr_median", "zcr_iqr"]),
    "mfcc": ("_mfcc.csv", [f"mfcc_c{i}_mean" for i in range(13)]),
    "cpps": ("_cpps.csv", ["cpps"]),
}
FEATURE_COLS = [c for _, cols in FAMILY_SPEC.values() for c in cols]   # 41


def load_features(parent, prefix, id_re):
    merged = None
    for fam, (suffix, cols) in FAMILY_SPEC.items():
        rows = []
        for p in sorted(glob.glob(f"{parent}/{prefix}_{fam}_metadata/*{suffix}")):
            m = id_re.search(os.path.basename(p))
            if not m:
                continue
            head = pd.read_csv(p, nrows=1)
            rows.append({"id": m.group(1), **{c: head.iloc[0][c] for c in cols if c in head.columns}})
        if not rows:
            continue                       # family folder empty / no matching CSVs -> skip
        df = pd.DataFrame(rows)
        merged = df if merged is None else merged.merge(df, on="id", how="outer")
    return merged


# ---- score / timestamp loaders (return id, ts, madrs_total, vas_anxiety, vas_depression, hamd_total) ----
def scores_from_csv(csv, ts_cols):
    d = pd.read_csv(csv)
    d["id"] = d["record_id"].astype(str)
    ts = None
    for c in ts_cols:
        if c in d.columns:
            cur = pd.to_datetime(d[c], errors="coerce")
            ts = cur if ts is None else ts.fillna(cur)
    d["ts"] = ts
    for c in ["madrs_total", "vas_anxiety", "vas_depression", "hamd_total", "hamd_q1", "hamd_q2", "hamd_q3", "hamd_q4", "hamd_q5", "hamd_q6"]:
        d[c] = pd.to_numeric(d[c], errors="coerce") if c in d.columns else np.nan
    return d[["id", "ts", "madrs_total", "vas_anxiety", "vas_depression", "hamd_total", "hamd_q1", "hamd_q2", "hamd_q3", "hamd_q4", "hamd_q5", "hamd_q6"]].dropna(subset=["ts"]).drop_duplicates("id")


def scores_from_xlsx(xlsx, sheet, ts_col):
    d = pd.read_excel(xlsx, sheet_name=sheet)
    d = d[d["Filename"].notna()].copy()
    d["id"] = d["Filename"].astype(str).str.extract(r"^(\d+)")[0]
    d["ts"] = pd.to_datetime(d[ts_col], errors="coerce")
    for c in ["madrs_total", "vas_anxiety", "vas_depression", "hamd_total", "hamd_q1", "hamd_q2", "hamd_q3", "hamd_q4", "hamd_q5", "hamd_q6"]:
        d[c] = pd.to_numeric(d[c], errors="coerce") if c in d.columns else np.nan
    return d[["id", "ts", "madrs_total", "vas_anxiety", "vas_depression", "hamd_total", "hamd_q1", "hamd_q2", "hamd_q3", "hamd_q4", "hamd_q5", "hamd_q6"]].dropna(subset=["id", "ts"]).drop_duplicates("id")


def scores_from_clinician_audit(audit_csv, gap_tol=48.0):
    d = pd.read_csv(audit_csv)
    d["id"] = d["audio_id"].astype(str)
    d["ts"] = pd.to_datetime(d["rec_time"], errors="coerce")
    # scores from a survey > gap_tol hours away are unreliable -> NaN for coloring
    far = pd.to_numeric(d["gap_hours"], errors="coerce") > gap_tol
    for c in ["madrs_total", "vas_anxiety", "vas_depression", "hamd_total", "hamd_q1", "hamd_q2", "hamd_q3", "hamd_q4", "hamd_q5", "hamd_q6"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
        d.loc[far, c] = np.nan
    return d[["id", "ts", "madrs_total", "vas_anxiety", "vas_depression", "hamd_total", "hamd_q1", "hamd_q2", "hamd_q3", "hamd_q4", "hamd_q5", "hamd_q6"]].dropna(subset=["ts"]).drop_duplicates("id")


INDEX_ROW_RE = re.compile(
    r"<tr><td>(\d+)</td><td>audio</td><td>[^<]*</td>"
    r"<td><a href=\"documents/(\d+)_audio\.m4a\"[^>]*>[^<]+</a></td><td>([^<]+)</td></tr>")


def scores_from_pr09(csv, index_html):
    text = open(index_html).read()
    up = []
    for m in INDEX_ROW_RE.finditer(text):
        _, fn, uploaded = m.groups()
        up.append({"id": str(fn), "ts": datetime.strptime(html_lib.unescape(uploaded).strip(), "%m/%d/%Y %I:%M%p")})
    up = pd.DataFrame(up)
    d = pd.read_csv(csv)
    d = d[d.get("redcap_repeat_instrument") == "completion_pt"].copy()
    tcol = "start_local_timestamp" if "start_local_timestamp" in d.columns else "start_timestamp_local"
    d["survey_ts"] = pd.to_datetime(d[tcol].fillna(d["completion_pt_timestamp"]), errors="coerce")
    d = d.dropna(subset=["survey_ts"]).sort_values("survey_ts")
    for c in ["vas_anxiety", "vas_depression", "hamd_total", "hamd_q1", "hamd_q2", "hamd_q3", "hamd_q4", "hamd_q5", "hamd_q6"]:
        d[c] = pd.to_numeric(d[c], errors="coerce") if c in d.columns else np.nan
    d["madrs_total"] = np.nan
    merged = pd.merge_asof(up.sort_values("ts"), d.rename(columns={"survey_ts": "ts"}),
                           on="ts", direction="nearest")
    return merged[["id", "ts", "madrs_total", "vas_anxiety", "vas_depression", "hamd_total", "hamd_q1", "hamd_q2", "hamd_q3", "hamd_q4", "hamd_q5", "hamd_q6"]]


def run_patient(name, sources, out_parent, out_prefix, severity_col_label):
    feats_all, scores_all = [], []
    for parent, prefix, id_re, score_df, stage in sources:
        f = load_features(parent, prefix, id_re)
        f["stage"] = stage
        f["uid"] = stage + "_" + f["id"].astype(str)
        s = score_df.copy()
        s["uid"] = stage + "_" + s["id"].astype(str)
        feats_all.append(f)
        scores_all.append(s)
    F = pd.concat(feats_all, ignore_index=True)
    S = pd.concat(scores_all, ignore_index=True)[["uid", "ts", "madrs_total", "vas_anxiety", "vas_depression", "hamd_total"]]
    df = F.merge(S, on="uid", how="inner").dropna(subset=["ts"]).reset_index(drop=True)
    df = df.sort_values("ts").reset_index(drop=True)

    X = df.reindex(columns=FEATURE_COLS).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float).copy()
    # impute missing feature cells with column median (few Praat failures), then standardize
    col_med = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_med, inds[1])
    Xs = StandardScaler().fit_transform(X)

    n = len(df)
    perplexity = float(min(30, max(5, n // 4)))
    params = dict(n_components=2, perplexity=perplexity, learning_rate="auto",
                  init="pca", metric="euclidean", max_iter=1000, random_state=42)
    emb = TSNE(**params).fit_transform(Xs)
    df["tsne_x"], df["tsne_y"] = emb[:, 0], emb[:, 1]
    df["day_number"] = (df["ts"] - df["ts"].min()).dt.total_seconds() / 86400.0

    md = f"{out_parent}/{out_prefix}_tsne_metadata"
    pl = f"{out_parent}/{out_prefix}_tsne_plots"
    os.makedirs(md, exist_ok=True); os.makedirs(pl, exist_ok=True)
    keep = ["uid", "stage", "id", "ts", "day_number", "tsne_x", "tsne_y",
            "madrs_total", "vas_anxiety", "vas_depression", "hamd_total"]
    df[keep].to_csv(f"{md}/tsne_embedding.csv", index=False)
    json.dump({**params, "perplexity": perplexity, "n_samples": n,
               "n_features": len(FEATURE_COLS), "feature_columns": FEATURE_COLS,
               "preprocessing": "median-impute NaN then StandardScaler (z-score) per feature",
               "stages": df["stage"].value_counts().to_dict(),
               "severity_color": severity_col_label,
               "sklearn_TSNE": "sklearn.manifold.TSNE"}, open(f"{md}/tsne_params.json", "w"), indent=2, default=str)

    # ---- 4-panel figure ----
    sev_col = "madrs_total" if df["madrs_total"].notna().any() else "hamd_total"
    panels = [("day_number", "Day number", "viridis"),
              (sev_col, severity_col_label, "viridis_r"),
              ("vas_anxiety", "VAS-Anxiety", "viridis_r"),
              ("vas_depression", "VAS-Depression", "viridis_r")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for ax, (col, label, cmap) in zip(axes.ravel(), panels):
        c = pd.to_numeric(df[col], errors="coerce")
        ok = c.notna()
        ax.scatter(df.loc[~ok, "tsne_x"], df.loc[~ok, "tsne_y"], c="lightgray", s=18, alpha=0.5, label="no score")
        sc = ax.scatter(df.loc[ok, "tsne_x"], df.loc[ok, "tsne_y"], c=c[ok], s=22, cmap=cmap)
        fig.colorbar(sc, ax=ax, label=label)
        ax.set_title(f"t-SNE colored by {label}  (n={int(ok.sum())}/{n})", fontsize=11)
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    fig.suptitle(f"{name}: t-SNE of all acoustic features (perplexity={perplexity:.0f})", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{pl}/tsne_4panel.png", dpi=150)
    plt.close(fig)
    print(f"{name}: n={n} ({df['stage'].value_counts().to_dict()}) severity={sev_col} -> {pl}/tsne_4panel.png")


XLSX = f"{UD}/PR05 List of Video Filenames.xlsx"
S2P = f"{UD}/sub-PR05-stage-2_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots"
S2X = "sub-PR05_stage-2_audio-audiotype_preproc_spectral_gating_100_percent"
S3P = f"{UD}/sub-PR05-stage-3_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots"
S3X = "sub-PR05_stage-3_audio-audiotype_preproc_spectral_gating_100_percent"
CLP = f"{UD}/sub-PR05-clinician_scales_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots"
CLX = "sub-PR05_clinician_scales_audio-audiotype_preproc_spectral_gating_100_percent"
CL_AUDIT = f"{CLP}/{CLX}_hamd_correlation_datematched/score_match_audit.csv"

if __name__ == "__main__":
    # PR05 = Stage 2 + Stage 3 + Clinician (all data)
    run_patient("PR05 (Stage 2 + Stage 3 + Clinician)", [
        (S2P, S2X, SUBJECT_RE, scores_from_xlsx(XLSX, "Stage 2 AudioScore Match", "audio_task_timestamp"), "Stage2"),
        (S3P, S3X, SUBJECT_RE, scores_from_csv(f"{UD}/PR05Stage3_DATA_2026-06-07_1941.csv",
                                               ["audio_task_timestamp", "start_local_timestamp", "completion_pt_timestamp"]), "Stage3"),
        (CLP, CLX, CLINICIAN_RE, scores_from_clinician_audit(CL_AUDIT), "Clinician"),
    ], f"{UD}/PR05_tsne_all_data_metadata_and_plots", "PR05_tsne_all_data", "MADRS total")

    # PR08 (no MADRS -> HAM-D total as severity)
    P8 = f"{UD}/sub-PR08-pre-stage2_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots"
    X8 = "sub-PR08_pre-stage2_audio-audiotype_preproc_spectral_gating_100_percent"
    run_patient("PR08 Pre-Stage 2", [
        (P8, X8, SUBJECT_RE, scores_from_csv(f"{UD}/PR08PreStage2_DATA_2026-04-18_1616.csv",
                                            ["audio_task_timestamp", "start_timestamp_local", "completion_pt_timestamp"]), "Stage2"),
    ], P8, X8, "HAM-D total (no MADRS)")

    # PR09 (no MADRS -> HAM-D total; scores by nearest survey to upload time)
    P9 = f"{UD}/sub-PR09-stage-2_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots"
    X9 = "sub-PR09_stage-2_audio-audiotype_preproc_spectral_gating_100_percent"
    run_patient("PR09 Stage 2", [
        (P9, X9, SUBJECT_RE, scores_from_pr09(f"{UD}/PR09Stage2_DATA_2026-04-18_1544.csv",
            "/data_store2/resection/neuropsych_video/presidio/Stage2/PR09/home/Files_PR09Stage2_2026-04-18_1541/index.html"), "Stage2"),
    ], P9, X9, "HAM-D total (no MADRS)")
