import re
import html as html_lib
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

BASE = Path("/userdata/msharma")
SCORES_CSV = BASE / "chronic_pain_patient_data_1928757PerceptDBSFor_DATA_2026-07-22_1414.csv"
INDEX_HTML = Path("/data_store2/resection/neuropsych_video/presidio/Stage2/chronic_pain/RCS08/index.html")
RUN_PARENT = BASE / "sub-RCS08-stage-1_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots"
PREFIX = "sub-RCS08_stage-1_audio-audiotype_preproc_spectral_gating_100_percent"
OUT_DIR = RUN_PARENT / f"{PREFIX}_pain_mood_correlation"
DATASET_LABEL = "RCS08 Stage 1 (chronic pain)"

DAILY_TS = "stage_1_daily_surveys_vasnrsmpq_timestamp"
AUDIO_TS = "mood_vasaudio_recording_timestamp"
DAILY_COLS = ["pain_nrs_s1_daily", "pain_vas_s1_daily", "relief_vas_s1_daily", "mpq_s1_daily"]
WEEKLY_COL = "mood_vas_s1_weekly"
SCORE_COLS = DAILY_COLS + [WEEKLY_COL]
SCORE_LABELS = {
    "pain_nrs_s1_daily": "Pain NRS (daily)",
    "pain_vas_s1_daily": "Pain VAS (daily)",
    "relief_vas_s1_daily": "Pain relief VAS (daily)",
    "mpq_s1_daily": "McGill SF-MPQ (daily)",
    "mood_vas_s1_weekly": "Mood VAS (weekly)",
}

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
FEATURE_DIRS = {}
for _fam, (_suf, _cols) in FAMILY_SPEC.items():
    _d = RUN_PARENT / f"{PREFIX}_{_fam}_metadata"
    if _d.is_dir() and any(_d.glob(f"*{_suf}")):
        FEATURE_DIRS[_fam] = (_d, _suf, _cols)

SUBJECT_RE = re.compile(r"signal-preproc_(\d+)_")


def load_feature_summary(directory, suffix, cols):
    rows = []
    for csv_path in sorted(directory.glob(f"*{suffix}")):
        m = SUBJECT_RE.search(csv_path.name)
        if not m:
            continue
        head = pd.read_csv(csv_path, nrows=1)
        rows.append({"audio_id": m.group(1), **{c: head.iloc[0][c] for c in cols if c in head.columns}})
    return pd.DataFrame(rows)


def build_feature_table():
    merged = None
    for name, (d, suffix, cols) in FEATURE_DIRS.items():
        df = load_feature_summary(d, suffix, cols)
        df = df.rename(columns={c: f"{name}__{c}" for c in df.columns if c != "audio_id"})
        merged = df if merged is None else merged.merge(df, on="audio_id", how="outer")
    return merged


ROW_RE = re.compile(r"recording_(\d+)_audio_weekly\.m4a\"[^>]*>[^<]*</a></td><td>([^<]+)</td></tr>")


def parse_upload_datetimes():
    rows = []
    for m in ROW_RE.finditer(INDEX_HTML.read_text()):
        ts = datetime.strptime(html_lib.unescape(m.group(2)).strip(), "%m/%d/%Y %I:%M%p")
        rows.append({"audio_id": str(m.group(1)), "audio_uploaded": pd.Timestamp(ts)})
    return pd.DataFrame(rows)


def load_scores():
    d = pd.read_csv(SCORES_CSV, low_memory=False)
    dd = d[d[DAILY_TS].notna()].copy()
    dd["survey_ts"] = pd.to_datetime(dd[DAILY_TS], errors="coerce")
    dd = dd.dropna(subset=["survey_ts"])[["survey_ts"] + DAILY_COLS].sort_values("survey_ts").reset_index(drop=True)
    wk = d[d[WEEKLY_COL].notna()].copy()
    wk["survey_ts"] = pd.to_datetime(wk[AUDIO_TS], errors="coerce")
    wk = wk.dropna(subset=["survey_ts"])[["survey_ts", WEEKLY_COL]].sort_values("survey_ts").reset_index(drop=True)
    return dd, wk


def match_scores(uploads):
    dd, wk = load_scores()
    u = uploads.sort_values("audio_uploaded").reset_index(drop=True)
    m1 = pd.merge_asof(u, dd.rename(columns={"survey_ts": "audio_uploaded"}),
                       on="audio_uploaded", direction="nearest")
    m1["daily_gap_days"] = np.nan  # filled below via a second pass for audit
    # weekly
    m2 = pd.merge_asof(u, wk.rename(columns={"survey_ts": "audio_uploaded"}),
                       on="audio_uploaded", direction="nearest")
    out = m1.merge(m2[["audio_id", WEEKLY_COL]], on="audio_id", how="left")
    return out


def correlate(df, score_cols, feature_cols):
    out = []
    for s in score_cols:
        for f in feature_cols:
            sub = df[[s, f]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) < 3:
                continue
            rp, pp = pearsonr(sub[s], sub[f])
            rs, ps = spearmanr(sub[s], sub[f])
            out.append({"score": s, "feature": f, "n": len(sub),
                        "pearson_r": rp, "pearson_p": pp, "spearman_r": rs, "spearman_p": ps})
    return pd.DataFrame(out).sort_values("pearson_p")


def feature_label(col):
    if "__" not in col:
        return col
    family, metric = col.split("__", 1)
    if metric == family or metric == f"{family}_val":
        return family
    if metric.startswith(family):
        return metric
    return f"{family}_{metric}"


def _fmt_p(p):
    if np.isnan(p):
        return ""
    return f"{p:.0e}" if p < 1e-3 else f"{p:.3f}"


def plot_heatmap(df, score_cols, feature_cols, out_path):
    rmat = np.full((len(feature_cols), len(score_cols)), np.nan)
    pmat = np.full((len(feature_cols), len(score_cols)), np.nan)
    for i, f in enumerate(feature_cols):
        for j, s in enumerate(score_cols):
            sub = df[[s, f]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) >= 3:
                r, p = pearsonr(sub[s], sub[f]); rmat[i, j] = r; pmat[i, j] = p
    labels = [SCORE_LABELS.get(s, s) for s in score_cols]
    fig, ax = plt.subplots(figsize=(1.45 * len(score_cols) + 3, 0.55 * len(feature_cols) + 2))
    im = ax.imshow(rmat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(score_cols)), labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(feature_cols)), [feature_label(f) for f in feature_cols], fontsize=8)
    for i in range(rmat.shape[0]):
        for j in range(rmat.shape[1]):
            if not np.isnan(rmat[i, j]):
                star = "*" if pmat[i, j] < 0.05 else ""
                ax.text(j, i, f"{rmat[i, j]:.2f}{star}\np={_fmt_p(pmat[i, j])}", ha="center", va="center",
                        color="white" if abs(rmat[i, j]) > 0.5 else "black", fontsize=6)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title(f"{DATASET_LABEL}: pain/mood scores vs voice features (Pearson r; p below, * p<0.05)")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_scatter_grid(df, score, feature_cols, out_path):
    label = SCORE_LABELS.get(score, score)
    ncol = 4; nrow = int(np.ceil(len(feature_cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
    axes = np.array(axes).reshape(-1)
    for ax, f in zip(axes, feature_cols):
        fl = feature_label(f)
        sub = df[[score, f]].apply(pd.to_numeric, errors="coerce").dropna()
        ax.scatter(sub[score], sub[f], alpha=0.6, s=20)
        if len(sub) >= 3:
            r, p = pearsonr(sub[score], sub[f])
            z = np.polyfit(sub[score], sub[f], 1)
            xs = np.linspace(sub[score].min(), sub[score].max(), 50)
            ax.plot(xs, np.polyval(z, xs), color="red", lw=1)
            ax.set_title(f"{fl}\nr={r:.2f}, p={_fmt_p(p)}, n={len(sub)}", fontsize=9)
        else:
            ax.set_title(f"{fl}\n(n<3)", fontsize=9)
        ax.set_xlabel(label); ax.set_ylabel(fl, fontsize=8)
    for ax in axes[len(feature_cols):]:
        ax.axis("off")
    fig.suptitle(f"{DATASET_LABEL}: {label} vs voice features", fontsize=13)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    features = build_feature_table()
    features["audio_id"] = features["audio_id"].astype(str)
    uploads = parse_upload_datetimes()
    matched = match_scores(uploads)
    score_cols = [c for c in SCORE_COLS if c in matched.columns]
    df = features.merge(matched, on="audio_id", how="inner")
    print(f"Matched {len(df)} recordings (features={len(features)}, uploads={len(uploads)})")
    df.to_csv(OUT_DIR / "merged_scores_features.csv", index=False)
    feature_cols = [c for c in features.columns if c != "audio_id"]
    corr = correlate(df, score_cols, feature_cols)
    corr.to_csv(OUT_DIR / "correlations.csv", index=False)
    print("\nTop correlations by |pearson_r|:")
    print(corr.reindex(corr["pearson_r"].abs().sort_values(ascending=False).index).head(20).to_string(index=False))
    print("\nSignificant (p<0.05):", int((corr["pearson_p"] < 0.05).sum()), "of", len(corr), "feature-score pairs")
    plot_heatmap(df, score_cols, feature_cols, OUT_DIR / "correlation_heatmap.png")
    for s in score_cols:
        plot_scatter_grid(df, s, feature_cols, OUT_DIR / f"scatter_{s}_vs_features.png")
    print(f"\nOutputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
