#!/usr/bin/env python3
"""
mash every recording's feature csvs into one table (per patient/stage).

reads all the per-family metadata csvs pitch/loudness/f3/alpha/jitter/shimmer/
wpm/hnr/zcr/mfcc/cpps, grabs the stat columns and joins on recording id. the
column names are already unique across families so no prefixing needed.

put 3 things into each dataset's parent dir:
    <prefix>_all_features_metadata/    one csv per recording (everything)
    <prefix>_all_features.csv          all of it, one table
    <prefix>_all_features.pkl              ^ same thing, pickled df

full stat set (q1/q3, pitch slope + residual, wpm word_count/
dur/speech ...)
"""
import glob, os, re, html as html_lib
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd

UD = "/userdata/msharma"
NUM_RE = re.compile(r"signal-preproc_(?:wiener_)?(\d+)_")
CLIN_RE = re.compile(r"(GMT\d{8}-\d{6}_Recording)")
XLSX = f"{UD}/PR05 List of Video Filenames.xlsx"
INDEX9 = "/data_store2/resection/neuropsych_video/presidio/Stage2/PR09/home/Files_PR09Stage2_2026-04-18_1541/index.html"
IDX_RE = re.compile(r"<tr><td>(\d+)</td><td>audio</td><td>[^<]*</td><td><a href=\"documents/(\d+)_audio\.m4a\"[^>]*>[^<]+</a></td><td>([^<]+)</td></tr>")


def timestamps(spec, ids):
    """audio_id(str) -> recording Timestamp, per dataset. spec: ('csv',path,[cols]) |
    ('xlsx',path,sheet) | ('clinician',) | ('index',path)."""
    kind = spec[0]
    if kind == "csv":
        d = pd.read_csv(spec[1]); d["id"] = d["record_id"].astype(str)
        ts = None
        for c in spec[2]:
            if c in d.columns:
                cur = pd.to_datetime(d[c], errors="coerce"); ts = cur if ts is None else ts.fillna(cur)
        d["ts"] = ts
        return dict(zip(d["id"], d["ts"]))
    if kind == "xlsx":
        d = pd.read_excel(spec[1], sheet_name=spec[2]); d = d[d["Filename"].notna()]
        d["id"] = d["Filename"].astype(str).str.split("_").str[0]
        d["ts"] = pd.to_datetime(d["audio_task_timestamp"], errors="coerce")
        return dict(zip(d["id"], d["ts"]))
    if kind == "index":
        out = {}
        for m in IDX_RE.finditer(open(spec[1]).read()):
            out[str(m.group(2))] = pd.Timestamp(datetime.strptime(html_lib.unescape(m.group(3)).strip(), "%m/%d/%Y %I:%M%p"))
        return out
    if kind == "clinician":
        out = {}
        for cid in ids:
            g = re.search(r"GMT(\d{8})-(\d{6})", cid)
            out[cid] = pd.Timestamp(g.group(1) + g.group(2)).tz_localize("UTC").tz_convert(ZoneInfo("US/Pacific")).tz_localize(None)
        return out
    return {}

# family -> (filename suffix, feature columns to keep)  [column names are unique across families]
FAMILY_FEATURES = {
    "pitch": ("_pitches.csv", ["f0_lstsq_slope", "pitch_mean", "pitch_std", "pitch_median", "pitch_q1", "pitch_q3", "pitch_iqr", "lstsq_residual_std"]),
    "loudness": ("_loudness_in_db.csv", ["active_intensity_vals_mean", "intensity_std", "intensity_median", "intensity_q1", "intensity_q3", "intensity_iqr"]),
    "f3": ("_relative_energy_formant.csv", ["mean_rel_energy_f_i", "rel_energy_std", "rel_energy_median", "rel_energy_q1", "rel_energy_q3", "rel_energy_iqr"]),
    "alpha_ratio": ("_alpha_ratio.csv", ["alpha_ratio_mean", "alpha_ratio_std", "alpha_ratio_median", "alpha_ratio_q1", "alpha_ratio_q3", "alpha_ratio_iqr"]),
    "jitter": ("_jitter.csv", ["jitter_val"]),
    "shimmer": ("_shimmer_apqN.csv", ["shimmer_val"]),
    "wpm": ("_wpm.csv", ["word_count", "duration_sec", "speech_sec", "wpm", "wpm_articulation"]),
    "hnr": ("_hnr.csv", ["hnr_mean", "hnr_std", "hnr_median", "hnr_q1", "hnr_q3", "hnr_iqr"]),
    "zcr": ("_zcr.csv", ["zcr_mean", "zcr_std", "zcr_median", "zcr_q1", "zcr_q3", "zcr_iqr"]),
    "mfcc": ("_mfcc.csv", [f"mfcc_c{i}_mean" for i in range(13)]),
    "cpps": ("_cpps.csv", ["cpps"]),
}
FEATURE_ORDER = [c for _, cols in FAMILY_FEATURES.values() for c in cols]
SCORE_COLS = ["hamd_total", "hamd_q1", "hamd_q2", "hamd_q3", "hamd_q4", "hamd_q5",
              "hamd_q6", "vas_anxiety", "vas_depression", "madrs_total", "vas_lowenergy"]


def build(name, parent, prefix, id_re, ts_spec, corr_suffix):
    merged = None
    for fam, (suffix, cols) in FAMILY_FEATURES.items():
        rows = []
        for p in sorted(glob.glob(f"{parent}/{prefix}_{fam}_metadata/*{suffix}")):
            m = id_re.search(os.path.basename(p))
            if not m:
                continue
            head = pd.read_csv(p, nrows=1)
            rows.append({"audio_id": m.group(1), **{c: head.iloc[0][c] for c in cols if c in head.columns}})
        if not rows:
            print(f"  [{name}] WARNING: no files for family {fam}")
            continue
        df = pd.DataFrame(rows)
        merged = df if merged is None else merged.merge(df, on="audio_id", how="outer")
    if merged is None:
        print(f"  [{name}] no features found; skipping")
        return
    ts_map = timestamps(ts_spec, set(merged["audio_id"]))
    merged["timestamp"] = merged["audio_id"].map(ts_map)

    # merge symptom scores from this dataset's correlation output (by audio_id)
    score_cols = []
    msf = f"{parent}/{prefix}{corr_suffix}/merged_scores_features.csv"
    if os.path.exists(msf):
        s = pd.read_csv(msf)
        s["audio_id"] = s["audio_id"].astype(str)
        score_cols = [c for c in SCORE_COLS if c in s.columns]
        merged = merged.merge(s[["audio_id"] + score_cols].drop_duplicates("audio_id"), on="audio_id", how="left")
    else:
        print(f"  [{name}] WARNING: no scores file at {msf}")

    merged.insert(0, "patient_stage", name)
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    ordered = (["patient_stage", "audio_id", "timestamp"]
               + [c for c in FEATURE_ORDER if c in merged.columns]
               + score_cols)
    merged = merged[ordered]

    md = f"{parent}/{prefix}_all_features_metadata"
    os.makedirs(md, exist_ok=True)
    for _, r in merged.iterrows():
        r.to_frame().T.to_csv(f"{md}/{r['audio_id']}_all_features.csv", index=False)
    merged.to_csv(f"{parent}/{prefix}_all_features.csv", index=False)
    merged.to_pickle(f"{parent}/{prefix}_all_features.pkl")
    nfeat = len([c for c in FEATURE_ORDER if c in merged.columns])
    print(f"  [{name}] {len(merged)} recordings x {nfeat} features + {len(score_cols)} scores -> "
          f"{prefix}_all_features.csv + .pkl + {len(merged)} per-recording CSVs")


S3_CSV = f"{UD}/PR05Stage3_DATA_2026-06-07_1941.csv"
P8_CSV = f"{UD}/PR08PreStage2_DATA_2026-04-18_1616.csv"
SHEET = "Stage 2 AudioScore Match"
DATASETS = [
    ("PR05 Stage 2", f"{UD}/sub-PR05-stage-2_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots", "sub-PR05_stage-2_audio-audiotype_preproc_spectral_gating_100_percent", NUM_RE, ("xlsx", XLSX, SHEET), "_hamd_correlation"),
    ("PR05 Stage 3", f"{UD}/sub-PR05-stage-3_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots", "sub-PR05_stage-3_audio-audiotype_preproc_spectral_gating_100_percent", NUM_RE, ("csv", S3_CSV, ["audio_task_timestamp", "start_local_timestamp", "completion_pt_timestamp"]), "_hamd_correlation"),
    ("PR05 Stage 2 (wiener)", f"{UD}/sub-PR05-stage-2_audio-audiotype_preproc_wiener_filtering_metadata_and_plots", "sub-PR05_stage-2_audio-audiotype_preproc_wiener_filtering", NUM_RE, ("xlsx", XLSX, SHEET), "_hamd_correlation"),
    ("PR05 Clinician Scales", f"{UD}/sub-PR05-clinician_scales_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots", "sub-PR05_clinician_scales_audio-audiotype_preproc_spectral_gating_100_percent", CLIN_RE, ("clinician",), "_hamd_correlation_datematched"),
    ("PR08 Pre-Stage 2", f"{UD}/sub-PR08-pre-stage2_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots", "sub-PR08_pre-stage2_audio-audiotype_preproc_spectral_gating_100_percent", NUM_RE, ("csv", P8_CSV, ["audio_task_timestamp", "start_timestamp_local", "completion_pt_timestamp"]), "_hamd_correlation"),
    ("PR09 Stage 2", f"{UD}/sub-PR09-stage-2_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots", "sub-PR09_stage-2_audio-audiotype_preproc_spectral_gating_100_percent", NUM_RE, ("index", INDEX9), "_hamd_correlation"),
]

if __name__ == "__main__":
    for name, parent, prefix, id_re, ts_spec, corr_suffix in DATASETS:
        build(name, parent, prefix, id_re, ts_spec, corr_suffix)
