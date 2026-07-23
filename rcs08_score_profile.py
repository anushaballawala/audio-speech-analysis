import pandas as pd, numpy as np
CSV = "/userdata/msharma/chronic_pain_patient_data_1928757PerceptDBSFor_DATA_2026-07-22_1414.csv"
d = pd.read_csv(CSV, low_memory=False)
print("rows:", len(d))

# rows that are the weekly audio recordings
aud = d[d["audio_weekly"].notna()]
print(f"\naudio_weekly non-null rows (recordings in the score file): {len(aud)}")
print("  mood_vas_s1_weekly non-null among those:", aud["mood_vas_s1_weekly"].notna().sum())
print("  sample audio_weekly values:", aud["audio_weekly"].head(3).tolist())
print("  mood_vasaudio_recording_timestamp range:",
      pd.to_datetime(aud["mood_vasaudio_recording_timestamp"], errors="coerce").min(),
      "->", pd.to_datetime(aud["mood_vasaudio_recording_timestamp"], errors="coerce").max())

# candidate score columns: (label, value col, timestamp col)
cands = [
    ("HAM-D total", "hamd_score", "hamilton_selfrating_scale_for_depression_hamd_timestamp"),
    ("HAM-D q1", "hamd_q1", "hamilton_selfrating_scale_for_depression_hamd_timestamp"),
    ("pain NRS (nrsvas)", "pain_nrs", "nrsvas_timestamp"),
    ("pain VAS (nrsvas)", "pain_vas", "nrsvas_timestamp"),
    ("depression VAS (nrsvas)", "depression_vas", "nrsvas_timestamp"),
    ("anxiety VAS (nrsvas)", "anxiety_vas", "nrsvas_timestamp"),
    ("unpleasantness VAS", "unpleasantness_vas", "nrsvas_timestamp"),
    ("mood VAS weekly (w/ audio)", "mood_vas_s1_weekly", "mood_vasaudio_recording_timestamp"),
    ("mood VAS (pain_vas_mood)", "mood_vas", "pain_vas_mood_timestamp"),
    ("BPI pain severity", "pain_severity_score_bpi", "brief_pain_inventory_timestamp"),
    ("BPI pain interference", "pain_interference_score_bpi", "brief_pain_inventory_timestamp"),
    ("pain catastrophizing total", "total_score_pc", "pain_catastrophizing_scale_timestamp"),
    ("SF-MPQ sum", "sf_mpq_sum", "stage_0_long_nrsvasmpq_timestamp"),
]
print(f"\n{'score':32s} {'n_nonnull':>9} {'date_min':>12} {'date_max':>12}")
for lbl, col, tcol in cands:
    if col not in d.columns:
        print(f"{lbl:32s}  MISSING COLUMN {col}"); continue
    sub = d[d[col].notna()]
    ts = pd.to_datetime(sub[tcol], errors="coerce") if tcol in d.columns else pd.Series([], dtype="datetime64[ns]")
    dmin = ts.min().date() if ts.notna().any() else "NaT"
    dmax = ts.max().date() if ts.notna().any() else "NaT"
    print(f"{lbl:32s} {len(sub):9d} {str(dmin):>12} {str(dmax):>12}")
