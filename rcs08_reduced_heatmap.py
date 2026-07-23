import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import ceil

CORR_DIR = ("/userdata/msharma/sub-RCS08-stage-1_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots/"
            "sub-RCS08_stage-1_audio-audiotype_preproc_spectral_gating_100_percent_pain_mood_correlation")
DATASET_LABEL = "RCS08 Stage 1 (chronic pain)"
SCORE_ORDER = ["pain_nrs_s1_daily", "pain_vas_s1_daily", "relief_vas_s1_daily", "mpq_s1_daily", "mood_vas_s1_weekly"]
SCORE_LABELS = {
    "pain_nrs_s1_daily": "Pain NRS (daily)",
    "pain_vas_s1_daily": "Pain VAS (daily)",
    "relief_vas_s1_daily": "Pain relief VAS (daily)",
    "mpq_s1_daily": "McGill SF-MPQ (daily)",
    "mood_vas_s1_weekly": "Mood VAS (weekly)",
}


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


c = pd.read_csv(f"{CORR_DIR}/correlations.csv")
R = c.pivot_table(index="feature", columns="score", values="pearson_r")
P = c.pivot_table(index="feature", columns="score", values="pearson_p")
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
ax.set_title(f"{DATASET_LABEL}: reduced heatmap — top third of features by |r| "
             f"({k} of {len(strength)}); p below, * p<0.05")
fig.tight_layout()
out = f"{CORR_DIR}/reduced_heatmap.png"
fig.savefig(out, dpi=150); plt.close(fig)
print(f"top {k}/{len(strength)} features -> {out}")
