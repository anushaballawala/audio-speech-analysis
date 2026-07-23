"""
reduced_heatmap.py -- per patient/stage, a heatmap of only the TOP THIRD of
features (the ones that correlate most strongly, by max |pearson r| across that
dataset's scores). reads the correlations.csv already written by each
hamd_feature_correlation_* run, so no re-correlating. writes reduced_heatmap.png
next to the full correlation_heatmap.png.
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import ceil

UD = "/userdata/msharma"
SCORE_ORDER = ["hamd_total", "hamd_q1", "hamd_q2", "hamd_q3", "hamd_q4", "hamd_q5",
               "hamd_q6", "vas_anxiety", "vas_depression", "madrs_total", "vas_lowenergy"]
SCORE_LABELS = {"hamd_total": "HAM-D total", "hamd_q1": "1. Depressed mood",
                "hamd_q2": "2. Self-esteem and guilt", "hamd_q3": "3. Social interaction and interests",
                "hamd_q4": "4. Psychomotor retardation", "hamd_q5": "5. Anxiety", "hamd_q6": "6. Somatic symptoms",
                "vas_anxiety": "VAS anxiety", "vas_depression": "VAS depression",
                "madrs_total": "MADRS total", "vas_lowenergy": "VAS low energy"}


def feature_label(col):
    if "__" not in col:
        return col
    fam, met = col.split("__", 1)
    if met == fam or met == f"{fam}_val":
        return fam
    return met if met.startswith(fam) else f"{fam}_{met}"


def _fmt_p(p):
    if pd.isna(p):
        return ""
    return f"{p:.0e}" if p < 1e-3 else f"{p:.3f}"


def make(label, corr_dir):
    c = pd.read_csv(f"{corr_dir}/correlations.csv")
    R = c.pivot_table(index="feature", columns="score", values="pearson_r")
    P = c.pivot_table(index="feature", columns="score", values="pearson_p")
    # rank features by strongest |r| across scores; keep top third
    strength = R.abs().max(axis=1)
    k = max(1, ceil(len(strength) / 3))
    top = strength.sort_values(ascending=False).head(k).index.tolist()
    R, P = R.loc[top], P.loc[top]
    scores = [s for s in SCORE_ORDER if s in R.columns] + [s for s in R.columns if s not in SCORE_ORDER]
    R, P = R[scores], P[scores]

    rmat, pmat = R.to_numpy(float), P.to_numpy(float)
    fig, ax = plt.subplots(figsize=(1.45 * len(scores) + 3, 0.55 * len(top) + 2))
    im = ax.imshow(rmat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(scores)), [SCORE_LABELS.get(s, s) for s in scores], rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(top)), [feature_label(f) for f in top], fontsize=8)
    for i in range(rmat.shape[0]):
        for j in range(rmat.shape[1]):
            if not np.isnan(rmat[i, j]):
                star = "*" if pmat[i, j] < 0.05 else ""
                ax.text(j, i, f"{rmat[i, j]:.2f}{star}\np={_fmt_p(pmat[i, j])}", ha="center", va="center",
                        color="white" if abs(rmat[i, j]) > 0.5 else "black", fontsize=6)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title(f"{label}: reduced heatmap — top third of features by |r| "
                 f"({k} of {len(strength)}); p below, * p<0.05")
    fig.tight_layout()
    out = f"{corr_dir}/reduced_heatmap.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[{label}] top {k}/{len(strength)} features -> {out}")


DATASETS = [
    ("PR05 Stage 2", "sub-PR05-stage-2_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots/sub-PR05_stage-2_audio-audiotype_preproc_spectral_gating_100_percent_hamd_correlation"),
    ("PR05 Stage 3", "sub-PR05-stage-3_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots/sub-PR05_stage-3_audio-audiotype_preproc_spectral_gating_100_percent_hamd_correlation"),
    ("PR05 Stage 2 (wiener)", "sub-PR05-stage-2_audio-audiotype_preproc_wiener_filtering_metadata_and_plots/sub-PR05_stage-2_audio-audiotype_preproc_wiener_filtering_hamd_correlation"),
    ("PR05 Clinician Scales", "sub-PR05-clinician_scales_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots/sub-PR05_clinician_scales_audio-audiotype_preproc_spectral_gating_100_percent_hamd_correlation_datematched"),
    ("PR08 Pre-Stage 2", "sub-PR08-pre-stage2_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots/sub-PR08_pre-stage2_audio-audiotype_preproc_spectral_gating_100_percent_hamd_correlation"),
    ("PR09 Stage 2", "sub-PR09-stage-2_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots/sub-PR09_stage-2_audio-audiotype_preproc_spectral_gating_100_percent_hamd_correlation"),
]

if __name__ == "__main__":
    for label, cd in DATASETS:
        make(label, f"{UD}/{cd}")
