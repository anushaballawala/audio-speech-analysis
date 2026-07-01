#!/usr/bin/env python3
"""
Per-patient heatmap of pitch + MFCC feature VALUES across recordings (time-ordered).
"""

import re, glob, os, html as html_lib
from datetime import datetime
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

UD = "/userdata/msharma"
SUBJECT_RE = re.compile(r"signal-preproc_(?:wiener_)?(\d+)_")
PITCH_COLS = ["pitch_mean", "pitch_std", "pitch_median", "pitch_iqr"]
MFCC_COLS = [f"mfcc_c{i}_mean" for i in range(13)]

INDEX_ROW_RE = re.compile(
    r"<tr><td>(\d+)</td><td>audio</td><td>[^<]*</td>"
    r"<td><a href=\"documents/(\d+)_audio\.m4a\"[^>]*>[^<]+</a></td>"
    r"<td>([^<]+)</td></tr>")


CLINICIAN_RE = re.compile(r"(GMT\d{8}-\d{6}_Recording)")


def load_family(parent, prefix, fam, suffix, cols, id_re=SUBJECT_RE):
    rows = []
    for p in sorted(glob.glob(f"{parent}/{prefix}_{fam}_metadata/*{suffix}")):
        m = id_re.search(os.path.basename(p))
        if not m:
            continue
        head = pd.read_csv(p, nrows=1)
        rows.append({"id": m.group(1), **{c: head.iloc[0][c] for c in cols if c in head.columns}})
    return pd.DataFrame(rows)


def timestamps_from_csv(csv, ts_cols):
    d = pd.read_csv(csv)
    d["id"] = d["record_id"].astype(str)
    ts = None
    for c in ts_cols:
        if c in d.columns:
            cur = pd.to_datetime(d[c], errors="coerce")
            ts = cur if ts is None else ts.fillna(cur)
    d["ts"] = ts
    return d[["id", "ts"]].dropna(subset=["ts"])


def timestamps_from_index(path):
    text = open(path).read()
    rows = []
    for m in INDEX_ROW_RE.finditer(text):
        _, file_num, uploaded = m.groups()
        ts = datetime.strptime(html_lib.unescape(uploaded).strip(), "%m/%d/%Y %I:%M%p")
        rows.append({"id": str(file_num), "ts": ts})
    return pd.DataFrame(rows)


def load_source(parent, prefix, ts_df, stage, id_re=SUBJECT_RE):
    pitch = load_family(parent, prefix, "pitch", "_pitches.csv", PITCH_COLS, id_re)
    mfcc = load_family(parent, prefix, "mfcc", "_mfcc.csv", MFCC_COLS, id_re)
    feats = pitch.merge(mfcc, on="id", how="inner").merge(ts_df, on="id", how="inner")
    feats["stage"] = stage
    return feats


def timestamps_clinician(parent, prefix):
    """Clinician recording time = GMT filename (UTC) -> US/Pacific local (DST-aware)."""
    from zoneinfo import ZoneInfo
    ids = set()
    for p in glob.glob(f"{parent}/{prefix}_pitch_metadata/*_pitches.csv"):
        m = CLINICIAN_RE.search(os.path.basename(p))
        if m:
            ids.add(m.group(1))
    rows = []
    for cid in ids:
        g = re.search(r"GMT(\d{8})-(\d{6})", cid)
        utc = pd.Timestamp(g.group(1) + g.group(2)).tz_localize("UTC")
        rows.append({"id": cid, "ts": utc.tz_convert(ZoneInfo("US/Pacific")).tz_localize(None)})
    return pd.DataFrame(rows)


def plot_heatmap(patient, feats, out_png):
    feats = feats.sort_values("ts").reset_index(drop=True)
    n = len(feats)
    rows = PITCH_COLS + MFCC_COLS
    M = feats[rows].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float).T  # features x recordings

    # z-score each feature against the TOTAL (pooled) mean/SD across all recordings.
    mu = np.nanmean(M, axis=1, keepdims=True)
    sd = np.nanstd(M, axis=1, keepdims=True)
    Z = np.where(sd > 0, (M - mu) / sd, 0.0)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.10), 7))
    im = ax.imshow(Z, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    ax.axhline(len(PITCH_COLS) - 0.5, color="black", lw=1.2)  # divide pitch / MFCC blocks
    step = max(1, n // 12)
    idx = list(range(0, n, step))
    ax.set_xticks(idx, [feats["ts"].iloc[i].strftime("%Y-%m-%d") for i in idx], rotation=90, fontsize=7)
    counts = feats["stage"].value_counts().to_dict()
    cstr = ", ".join(f"{k}={counts[k]}" for k in counts)
    ax.set_xlabel(f"Recording (ordered by date) — n={n} ({cstr})")
    ax.set_title(f"{patient}: pitch & MFCC feature values across recordings\n"
                 f"(color = per-feature z-score vs the total/pooled mean across all recordings)")
    fig.colorbar(im, ax=ax, label="z-score (vs total mean)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"{patient}: {n} recordings ({cstr}) -> {out_png}")


def timestamps_from_xlsx(xlsx, sheet, ts_col):
    df = pd.read_excel(xlsx, sheet_name=sheet)
    df = df[df["Filename"].notna()].copy()
    df["id"] = df["Filename"].astype(str).str.extract(r"^(\d+)")[0]
    df["ts"] = pd.to_datetime(df[ts_col], errors="coerce")
    return df[["id", "ts"]].dropna(subset=["id", "ts"]).drop_duplicates("id")


XLSX = f"{UD}/PR05 List of Video Filenames.xlsx"

if __name__ == "__main__":
    out = f"{UD}/feature_time_heatmaps"
    os.makedirs(out, exist_ok=True)

    # ---- PR05: ALL data = Stage 2 (dates from XLSX) + Stage 3 (dates from CSV) ----
    pr05_s2 = load_source(
        f"{UD}/sub-PR05-stage-2_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots",
        "sub-PR05_stage-2_audio-audiotype_preproc_spectral_gating_100_percent",
        timestamps_from_xlsx(XLSX, "Stage 2 AudioScore Match", "audio_task_timestamp"), "Stage2")
    pr05_s3 = load_source(
        f"{UD}/sub-PR05-stage-3_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots",
        "sub-PR05_stage-3_audio-audiotype_preproc_spectral_gating_100_percent",
        timestamps_from_csv(f"{UD}/PR05Stage3_DATA_2026-06-07_1941.csv",
                            ["audio_task_timestamp", "start_local_timestamp", "completion_pt_timestamp"]), "Stage3")
    pr05_clin_parent = f"{UD}/sub-PR05-clinician_scales_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots"
    pr05_clin_prefix = "sub-PR05_clinician_scales_audio-audiotype_preproc_spectral_gating_100_percent"
    pr05_clin = load_source(pr05_clin_parent, pr05_clin_prefix,
                            timestamps_clinician(pr05_clin_parent, pr05_clin_prefix),
                            "Clinician", id_re=CLINICIAN_RE)
    pr05 = pd.concat([pr05_s2, pr05_s3, pr05_clin], ignore_index=True)
    plot_heatmap("PR05 (Stage 2 + Stage 3 + Clinician)", pr05, f"{out}/PR05_pitch_mfcc_time_heatmap.png")

    # ---- PR08 / PR09: single Stage-2 sources ----
    pr08 = load_source(
        f"{UD}/sub-PR08-stage-2_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots",
        "sub-PR08_stage-2_audio-audiotype_preproc_spectral_gating_100_percent",
        timestamps_from_csv(f"{UD}/PR08PreStage2_DATA_2026-04-18_1616.csv",
                            ["audio_task_timestamp", "start_timestamp_local", "completion_pt_timestamp"]), "Stage2")
    plot_heatmap("PR08", pr08, f"{out}/PR08_pitch_mfcc_time_heatmap.png")

    pr09 = load_source(
        f"{UD}/sub-PR09-stage-2_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots",
        "sub-PR09_stage-2_audio-audiotype_preproc_spectral_gating_100_percent",
        timestamps_from_index("/data_store2/resection/neuropsych_video/presidio/Stage2/PR09/home/Files_PR09Stage2_2026-04-18_1541/index.html"), "Stage2")
    plot_heatmap("PR09", pr09, f"{out}/PR09_pitch_mfcc_time_heatmap.png")
