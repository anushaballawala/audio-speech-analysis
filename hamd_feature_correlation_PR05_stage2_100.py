import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

BASE = Path("/userdata/msharma")
XLSX = BASE / "PR05 List of Video Filenames.xlsx"
SHEET = "Stage 2 AudioScore Match"
PREFIX = "sub-PR05_stage-2_audio-audiotype_preproc_spectral_gating_100_percent"
RUN_PARENT = BASE / "sub-PR05-stage-2_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots"
OUT_DIR = RUN_PARENT / "sub-PR05_stage-2_audio-audiotype_preproc_spectral_gating_100_percent_hamd_correlation"

# Score set: HAM-D total + its six individual clinician items, plus the VAS and
# MADRS totals. Any column not present in this CSV (e.g. madrs_score) is
# skipped by the `in scores.columns` filter.
SCORE_COLS = ["hamd_total", "vas_anxiety", "vas_depression", "madrs_total", "madrs_score"]

# Display labels for the six-item clinician HAM-D responses (per the standard
# 6-item structure) and the other scores, used on heatmap/scatter axes.
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
    "madrs_total": "MADRS total",
    "madrs_score": "MADRS score",
}

# Family spec; FEATURE_DIRS is built from PREFIX and filtered to families that
# actually have extracted CSVs, so the same code serves every Stage-2 variant.
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

# Matches  sub-PR05_stage-3_audio_signal-preproc_<id>_<metric>.csv
SUBJECT_RE = re.compile(r"signal-preproc_(?:wiener_)?(\d+)_")


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


def load_scores() -> pd.DataFrame:
    # Stage-2 at-home audio is matched to scores via the validated XLSX lookup
    # sheet (the audio file number is the leading integer of Filename); this is
    # the same join the wiener Stage-2 correlation used (214 subjects).
    df = pd.read_excel(XLSX, sheet_name=SHEET)
    df = df[df["Filename"].notna()].copy()
    df["audio_id"] = df["Filename"].astype(str).str.extract(r"^(\d+)")[0]
    keep = ["audio_id"] + [c for c in SCORE_COLS if c in df.columns]
    df = df[keep].dropna(subset=["audio_id"])
    df = df.groupby("audio_id", as_index=False).first()
    return df


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
        if len(sub) >= 3:
            r, p = pearsonr(sub[score], sub[f])
            z = np.polyfit(sub[score], sub[f], 1)
            xs = np.linspace(sub[score].min(), sub[score].max(), 50)
            ax.plot(xs, np.polyval(z, xs), color="red", lw=1)
            ax.set_title(f"{flabel}\nr={r:.2f}, p={_fmt_p(p)}, n={len(sub)}", fontsize=9)
        else:
            ax.set_title(f"{flabel}\n(n<3)", fontsize=9)
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
