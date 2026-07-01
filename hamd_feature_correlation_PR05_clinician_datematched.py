import re
import os
import glob
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

BASE = Path("/userdata/msharma")
# Two at-home self-report survey exports (2026-06-29) that together span the
# clinician-recording date range. Scores are matched to each clinician recording
# by NEAREST timestamp (see load_scores), not by a filename key.
STAGE2_CSV = BASE / "PR05Stage2_DATA_2026-06-29_1836.csv"
STAGE3_CSV = BASE / "PR05Stage3_DATA_LABELS_2026-06-29_1844.csv"
HAMD_MAP = BASE / "hamd_label_map.json"      # Stage-3 HAMD item label -> 0..4 (verified lossless)
PREPROC_DIR = "/data_store2/resection/neuropsych_video/presidio/Stage2/ClinicianScales/PR05/PR05_clinician_scales_audio_preproc_spectral_gating_100_percent"
MATCH_TOLERANCE_H = 48.0                      # drop recordings whose nearest survey is > this many hours away

RUN_PARENT = BASE / "sub-PR05-clinician_scales_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots"
OUT_DIR = RUN_PARENT / "sub-PR05_clinician_scales_audio-audiotype_preproc_spectral_gating_100_percent_hamd_correlation_datematched"

# Score set: HAM-D-6 total + its six individual items, plus VAS anxiety/depression
# and (self-report) MADRS total. These are the columns available numerically in
# BOTH survey exports (Stage-3 HAMD items are derived from text labels via HAMD_MAP).
SCORE_COLS = [
    "hamd_total",
    "hamd_q1", "hamd_q2", "hamd_q3", "hamd_q4", "hamd_q5", "hamd_q6",
    "vas_anxiety", "vas_depression", "madrs_total",
]

# Display labels for the six HAM-D-6 self-report items (standard 6-item structure)
# and the other scores, used on heatmap/scatter axes.
SCORE_LABELS = {
    "hamd_total": "HAM-D total",
    "hamd_q1": "1. Depressed mood",
    "hamd_q2": "2. Self-esteem and guilt",
    "hamd_q3": "3. Social interaction and interests",
    "hamd_q4": "4. Psychomotor retardation",
    "hamd_q5": "5. Anxiety",
    "hamd_q6": "6. Somatic symptoms",
    "vas_anxiety": "VAS anxiety",
    "vas_depression": "VAS depression",
    "madrs_total": "MADRS total (self-report)",
}

FEATURE_DIRS = {
    "pitch": (
        RUN_PARENT / "sub-PR05_clinician_scales_audio-audiotype_preproc_spectral_gating_100_percent_pitch_metadata",
        "_pitches.csv",
        ["pitch_mean", "pitch_std", "pitch_median", "pitch_iqr"],
    ),
    "loudness": (
        RUN_PARENT / "sub-PR05_clinician_scales_audio-audiotype_preproc_spectral_gating_100_percent_loudness_metadata",
        "_loudness_in_db.csv",
        ["active_intensity_vals_mean", "intensity_std", "intensity_median", "intensity_iqr"],
    ),
    "f3": (
        RUN_PARENT / "sub-PR05_clinician_scales_audio-audiotype_preproc_spectral_gating_100_percent_f3_metadata",
        "_relative_energy_formant.csv",
        ["mean_rel_energy_f_i", "rel_energy_std", "rel_energy_median", "rel_energy_iqr"],
    ),
    "alpha_ratio": (
        RUN_PARENT / "sub-PR05_clinician_scales_audio-audiotype_preproc_spectral_gating_100_percent_alpha_ratio_metadata",
        "_alpha_ratio.csv",
        ["alpha_ratio_mean", "alpha_ratio_std", "alpha_ratio_median", "alpha_ratio_iqr"],
    ),
    "jitter": (
        RUN_PARENT / "sub-PR05_clinician_scales_audio-audiotype_preproc_spectral_gating_100_percent_jitter_metadata",
        "_jitter.csv",
        ["jitter_val"],
    ),
    "shimmer": (
        RUN_PARENT / "sub-PR05_clinician_scales_audio-audiotype_preproc_spectral_gating_100_percent_shimmer_metadata",
        "_shimmer_apqN.csv",
        ["shimmer_val"],
    ),
    "wpm": (
        RUN_PARENT / "sub-PR05_clinician_scales_audio-audiotype_preproc_spectral_gating_100_percent_wpm_metadata",
        "_wpm.csv",
        ["wpm"],
    ),
    "hnr": (
        RUN_PARENT / "sub-PR05_clinician_scales_audio-audiotype_preproc_spectral_gating_100_percent_hnr_metadata",
        "_hnr.csv",
        ["hnr_mean", "hnr_std", "hnr_median", "hnr_iqr"],
    ),
    "zcr": (
        RUN_PARENT / "sub-PR05_clinician_scales_audio-audiotype_preproc_spectral_gating_100_percent_zcr_metadata",
        "_zcr.csv",
        ["zcr_mean", "zcr_std", "zcr_median", "zcr_iqr"],
    ),
    "mfcc": (
        RUN_PARENT / "sub-PR05_clinician_scales_audio-audiotype_preproc_spectral_gating_100_percent_mfcc_metadata",
        "_mfcc.csv",
        ["mfcc_c0_mean", "mfcc_c1_mean", "mfcc_c2_mean", "mfcc_c3_mean", "mfcc_c4_mean", "mfcc_c5_mean", "mfcc_c6_mean", "mfcc_c7_mean", "mfcc_c8_mean", "mfcc_c9_mean", "mfcc_c10_mean", "mfcc_c11_mean", "mfcc_c12_mean"],
    ),
    "cpps": (
        RUN_PARENT / "sub-PR05_clinician_scales_audio-audiotype_preproc_spectral_gating_100_percent_cpps_metadata",
        "_cpps.csv",
        ["cpps"],
    ),
}

# Feature CSVs are named GMT<date>-<time>_Recording_isolated_preproc_<metric>.csv;
# the audio_id is the recording stem, which matches the CSV `Filename` (minus .m4a).
SUBJECT_RE = re.compile(r"(GMT\d{8}-\d{6}_Recording)")


def extract_id(path: Path) -> str | None:
    m = SUBJECT_RE.search(path.name)
    return m.group(1) if m else None


def load_feature_summary(directory: Path, suffix: str, cols: list[str]) -> pd.DataFrame:
    rows = []
    for csv_path in sorted(directory.glob(f"*{suffix}")):
        sid = extract_id(csv_path)
        if sid is None:
            continue
        head = pd.read_csv(csv_path, nrows=1)
        rows.append({"audio_id": sid, **{c: head.iloc[0][c] for c in cols if c in head.columns}})
    return pd.DataFrame(rows)


def build_feature_table() -> pd.DataFrame:
    merged = None
    for name, (d, suffix, cols) in FEATURE_DIRS.items():
        df = load_feature_summary(d, suffix, cols)
        df = df.rename(columns={c: f"{name}__{c}" for c in df.columns if c != "audio_id"})
        merged = df if merged is None else merged.merge(df, on="audio_id", how="outer")
    return merged


def _unified_survey_pool() -> pd.DataFrame:
    """Combine the Stage-2 (numeric) and Stage-3 (labelled) at-home survey exports
    into one timestamped table with a common numeric score schema."""
    hmap = json.loads(Path(HAMD_MAP).read_text())  # keys "<colidx>|||<label>" -> int

    s2 = pd.read_csv(STAGE2_CSV)
    a = pd.DataFrame({
        "time": pd.to_datetime(s2["start_local_timestamp"], errors="coerce"),
        "hamd_total": pd.to_numeric(s2["hamd_total"], errors="coerce"),
        **{f"hamd_q{k}": pd.to_numeric(s2[f"hamd_q{k}"], errors="coerce") for k in range(1, 7)},
        "vas_anxiety": pd.to_numeric(s2["vas_anxiety"], errors="coerce"),
        "vas_depression": pd.to_numeric(s2["vas_depression"], errors="coerce"),
        "madrs_total": pd.to_numeric(s2["madrs_total"], errors="coerce"),
        "source": "stage2",
    })

    s3 = pd.read_csv(STAGE3_CSV)
    # Stage-3 HAMD items live in columns 10-15 as text; map to 0..4 via HAMD_MAP.
    q = {f"hamd_q{k}": s3.iloc[:, ci].map(lambda l, ci=ci: hmap.get(f"{ci}|||{l}", np.nan))
         for k, ci in zip(range(1, 7), range(10, 16))}
    b = pd.DataFrame({
        "time": pd.to_datetime(s3["Date and time of assessment:"], errors="coerce"),
        "hamd_total": pd.to_numeric(s3.iloc[:, 16], errors="coerce"),   # "HAMD-6 score"
        **q,
        "vas_anxiety": pd.to_numeric(s3.iloc[:, 4], errors="coerce"),    # "Anxiety"
        "vas_depression": pd.to_numeric(s3.iloc[:, 6], errors="coerce"), # "Depression"
        "madrs_total": pd.to_numeric(s3.iloc[:, 29], errors="coerce"),   # "Total score"
        "source": "stage3",
    })

    pool = pd.concat([a, b], ignore_index=True).dropna(subset=["time"])
    return pool.sort_values("time").reset_index(drop=True)


def _recording_times() -> pd.DataFrame:
    """GMT filename (UTC) -> US/Pacific local time (DST-aware) for each recording."""
    rows = []
    for w in sorted(glob.glob(f"{PREPROC_DIR}/*.wav")):
        m = re.search(r"GMT(\d{8})-(\d{6})", os.path.basename(w))
        utc = pd.Timestamp(m.group(1) + m.group(2)).tz_localize("UTC")
        loc = utc.tz_convert(ZoneInfo("US/Pacific")).tz_localize(None)
        rows.append({"audio_id": os.path.basename(w).split("_Recording")[0] + "_Recording",
                     "rec_time": loc})
    return pd.DataFrame(rows).sort_values("rec_time").reset_index(drop=True)


def load_scores() -> pd.DataFrame:
    """Match each clinician recording to its NEAREST at-home survey by timestamp.

    Writes a full audit (recording time, matched survey time, gap, source, scores)
    to score_match_audit.csv. Recordings whose nearest survey is more than
    MATCH_TOLERANCE_H hours away (e.g. the 2026 recordings that post-date the last
    survey) are dropped from the correlation but retained in the audit.
    """
    pool = _unified_survey_pool()
    recs = _recording_times()
    m = pd.merge_asof(recs, pool, left_on="rec_time", right_on="time", direction="nearest")
    m["gap_hours"] = (m["rec_time"] - m["time"]).abs().dt.total_seconds() / 3600.0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit_cols = ["audio_id", "rec_time", "time", "gap_hours", "source"] + SCORE_COLS
    m[audit_cols].to_csv(OUT_DIR / "score_match_audit.csv", index=False)
    n_drop = int((m["gap_hours"] > MATCH_TOLERANCE_H).sum())
    print(f"survey pool={len(pool)} rows; recordings={len(recs)}; "
          f"matched within {MATCH_TOLERANCE_H:.0f}h={len(m) - n_drop}; dropped(gap too large)={n_drop}")

    keep = m[m["gap_hours"] <= MATCH_TOLERANCE_H].copy()
    return keep[["audio_id"] + SCORE_COLS]


def correlate(df: pd.DataFrame, score_cols: list[str], feature_cols: list[str]) -> pd.DataFrame:
    out = []
    for s in score_cols:
        for f in feature_cols:
            sub = df[[s, f]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) < 3:
                continue
            rp, pp = pearsonr(sub[s], sub[f])
            rs, ps = spearmanr(sub[s], sub[f])
            out.append({
                "score": s, "feature": f, "n": len(sub),
                "pearson_r": rp, "pearson_p": pp,
                "spearman_r": rs, "spearman_p": ps,
            })
    return pd.DataFrame(out).sort_values("pearson_p")


def _fmt_p(p: float) -> str:
    """Compact p-value: scientific for small p, 3-decimal otherwise."""
    if np.isnan(p):
        return ""
    if p < 1e-3:
        return f"{p:.0e}"
    return f"{p:.3f}"


def feature_label(col: str) -> str:
    """Display label for a `family__metric` column (CSV columns are unchanged).

    Single-metric families collapse to the family name (shimmer__shimmer_val ->
    'shimmer', wpm__wpm -> 'wpm'); multi-metric families drop the family prefix
    (pitch__pitch_std -> 'pitch_std', alpha_ratio__alpha_ratio_mean ->
    'alpha_ratio_mean').
    """
    if "__" not in col:
        return col
    family, metric = col.split("__", 1)
    if metric == family or metric == f"{family}_val":
        return family            # shimmer, jitter, wpm
    if metric.startswith(family):
        return metric            # pitch_std, alpha_ratio_mean (metric already names the family)
    return f"{family}_{metric}"  # f3_mean_rel_energy_f_i, loudness_active_intensity_vals_mean


def plot_heatmap(df: pd.DataFrame, score_cols: list[str], feature_cols: list[str], out_path: Path):
    rmat = np.full((len(feature_cols), len(score_cols)), np.nan)
    pmat = np.full((len(feature_cols), len(score_cols)), np.nan)
    for i, f in enumerate(feature_cols):
        for j, s in enumerate(score_cols):
            sub = df[[s, f]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) >= 3:
                r, p = pearsonr(sub[s], sub[f])
                rmat[i, j] = r
                pmat[i, j] = p

    labels = [SCORE_LABELS.get(s, s) for s in score_cols]
    fig, ax = plt.subplots(figsize=(1.45 * len(score_cols) + 3, 0.55 * len(feature_cols) + 2))
    im = ax.imshow(rmat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(score_cols)), labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(feature_cols)), [feature_label(f) for f in feature_cols], fontsize=8)
    # Each cell shows Pearson r (with significance star) over its p-value.
    for i in range(rmat.shape[0]):
        for j in range(rmat.shape[1]):
            if not np.isnan(rmat[i, j]):
                star = "*" if pmat[i, j] < 0.05 else ""
                txt = f"{rmat[i, j]:.2f}{star}\np={_fmt_p(pmat[i, j])}"
                ax.text(j, i, txt, ha="center", va="center",
                        color="white" if abs(rmat[i, j]) > 0.5 else "black", fontsize=6)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Clinical scores vs voice features (Pearson r; p below, * p<0.05)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scatter_grid(df: pd.DataFrame, score: str, feature_cols: list[str], out_path: Path):
    label = SCORE_LABELS.get(score, score)
    ncol = 4
    nrow = int(np.ceil(len(feature_cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
    axes = np.array(axes).reshape(-1)
    for ax, f in zip(axes, feature_cols):
        flabel = feature_label(f)
        sub = df[[score, f]].apply(pd.to_numeric, errors="coerce").dropna()
        ax.scatter(sub[score], sub[f], alpha=0.6, s=20)
        # only fit a trendline / correlation when both axes vary (constant columns
        # are common with the small clinician-scale n and break pearsonr/polyfit)
        if len(sub) >= 3 and sub[score].std() > 0 and sub[f].std() > 0:
            r, p = pearsonr(sub[score], sub[f])
            try:
                z = np.polyfit(sub[score], sub[f], 1)
                xs = np.linspace(sub[score].min(), sub[score].max(), 50)
                ax.plot(xs, np.polyval(z, xs), color="red", lw=1)
            except np.linalg.LinAlgError:
                pass
            ax.set_title(f"{flabel}\nr={r:.2f}, p={_fmt_p(p)}, n={len(sub)}", fontsize=9)
        else:
            ax.set_title(f"{flabel}\n(n<3 or constant)", fontsize=9)
        ax.set_xlabel(label)
        ax.set_ylabel(flabel, fontsize=8)
    for ax in axes[len(feature_cols):]:
        ax.axis("off")
    fig.suptitle(f"{label} vs voice features", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    features = build_feature_table()
    features["audio_id"] = features["audio_id"].astype(str)

    scores = load_scores()
    score_cols = [c for c in SCORE_COLS if c in scores.columns]

    df = scores.merge(features, on="audio_id", how="inner")
    print(f"Matched {len(df)} subjects (features={len(features)}, scores={len(scores)})")
    df.to_csv(OUT_DIR / "merged_scores_features.csv", index=False)

    feature_cols = [c for c in features.columns if c != "audio_id"]

    corr = correlate(df, score_cols, feature_cols)
    corr.to_csv(OUT_DIR / "correlations.csv", index=False)
    print("\nTop correlations by |pearson_r|:")
    print(corr.reindex(corr["pearson_r"].abs().sort_values(ascending=False).index).head(20).to_string(index=False))

    plot_heatmap(df, score_cols, feature_cols, OUT_DIR / "correlation_heatmap.png")
    for s in score_cols:
        plot_scatter_grid(df, s, feature_cols, OUT_DIR / f"scatter_{s}_vs_features.png")

    print(f"\nOutputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
