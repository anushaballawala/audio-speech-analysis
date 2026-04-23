"""HAM-D / VAS correlation + scatter plots vs voice features for PR08.

Audio files are numbered 1-69 (N_audio.m4a) and correspond directly to record_id in
PR08PreStage2_DATA_2026-04-18_1616.csv (one row per record, both audio_task timestamp
and completion_pt scores in the same row).

Usage:
    python hamd_feature_correlation_PR08.py
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

BASE = Path("/userdata/msharma")
SCORES_CSV = BASE / "PR08PreStage2_DATA_2026-04-18_1616.csv"
RUN_PARENT = BASE / "sub-PR08-stage-2_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots"
OUT_DIR = RUN_PARENT / "sub-PR08_stage-2_audio-audiotype_preproc_spectral_gating_100_percent_hamd_correlation"

SCORE_COLS = ["hamd_total", "vas_anxiety", "vas_depression", "vas_energy", "vas_lowenergy"]

FEATURE_DIRS = {
    "pitch": (
        RUN_PARENT / "sub-PR08_stage-2_audio-audiotype_preproc_spectral_gating_100_percent_pitch_metadata",
        "_pitches.csv",
        ["pitch_mean", "pitch_std", "pitch_median", "pitch_iqr"],
    ),
    "loudness": (
        RUN_PARENT / "sub-PR08_stage-2_audio-audiotype_preproc_spectral_gating_100_percent_loudness_metadata",
        "_loudness_in_db.csv",
        ["active_intensity_vals_mean", "intensity_std", "intensity_median", "intensity_iqr"],
    ),
    "f3": (
        RUN_PARENT / "sub-PR08_stage-2_audio-audiotype_preproc_spectral_gating_100_percent_f3_metadata",
        "_relative_energy_formant.csv",
        ["mean_rel_energy_f_i", "rel_energy_std", "rel_energy_median", "rel_energy_iqr"],
    ),
    "alpha_ratio": (
        RUN_PARENT / "sub-PR08_stage-2_audio-audiotype_preproc_spectral_gating_100_percent_alpha_ratio_metadata",
        "_alpha_ratio.csv",
        ["alpha_ratio_mean", "alpha_ratio_std", "alpha_ratio_median", "alpha_ratio_iqr"],
    ),
}

SUBJECT_RE = re.compile(r"signal-preproc_(\d+)_")


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


def load_scores_by_audio_id() -> pd.DataFrame:
    df = pd.read_csv(SCORES_CSV)
    df["audio_id"] = df["record_id"].astype(str)
    keep = ["audio_id"] + [c for c in SCORE_COLS if c in df.columns]
    return df[keep]


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


def plot_heatmap(df: pd.DataFrame, score_cols: list[str], feature_cols: list[str], out_path: Path):
    mat = np.full((len(feature_cols), len(score_cols)), np.nan)
    for i, f in enumerate(feature_cols):
        for j, s in enumerate(score_cols):
            sub = df[[s, f]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) >= 3:
                mat[i, j] = pearsonr(sub[s], sub[f])[0]

    fig, ax = plt.subplots(figsize=(1.2 * len(score_cols) + 3, 0.45 * len(feature_cols) + 2))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(score_cols)), score_cols, rotation=30, ha="right")
    ax.set_yticks(range(len(feature_cols)), feature_cols)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="white" if abs(mat[i, j]) > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Clinical scores vs voice features (Pearson r)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scatter_grid(df: pd.DataFrame, score: str, feature_cols: list[str], out_path: Path):
    ncol = 4
    nrow = int(np.ceil(len(feature_cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
    axes = np.array(axes).reshape(-1)
    for ax, f in zip(axes, feature_cols):
        sub = df[[score, f]].apply(pd.to_numeric, errors="coerce").dropna()
        ax.scatter(sub[score], sub[f], alpha=0.6, s=20)
        if len(sub) >= 3:
            r, p = pearsonr(sub[score], sub[f])
            z = np.polyfit(sub[score], sub[f], 1)
            xs = np.linspace(sub[score].min(), sub[score].max(), 50)
            ax.plot(xs, np.polyval(z, xs), color="red", lw=1)
            ax.set_title(f"{f}\nr={r:.2f}, p={p:.3f}, n={len(sub)}", fontsize=9)
        else:
            ax.set_title(f"{f}\n(n<3)", fontsize=9)
        ax.set_xlabel(score)
        ax.set_ylabel(f.split("__")[-1], fontsize=8)
    for ax in axes[len(feature_cols):]:
        ax.axis("off")
    fig.suptitle(f"{score} vs voice features", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    features = build_feature_table()
    features["audio_id"] = features["audio_id"].astype(str)

    scores = load_scores_by_audio_id()
    score_cols = [c for c in SCORE_COLS if c in scores.columns]

    df = features.merge(scores, on="audio_id", how="inner")
    print(f"Matched {len(df)} audio files (features={len(features)}, scores={len(scores)})")
    df.to_csv(OUT_DIR / "merged_scores_features.csv", index=False)

    feature_cols = [c for c in features.columns if c != "audio_id"]

    corr = correlate(df, score_cols, feature_cols)
    corr.to_csv(OUT_DIR / "correlations.csv", index=False)
    print("\nTop correlations by |pearson_r|:")
    print(corr.reindex(corr["pearson_r"].abs().sort_values(ascending=False).index).head(15).to_string(index=False))

    plot_heatmap(df, score_cols, feature_cols, OUT_DIR / "correlation_heatmap.png")
    for s in score_cols:
        plot_scatter_grid(df, s, feature_cols, OUT_DIR / f"scatter_{s}_vs_features.png")

    print(f"\nOutputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
