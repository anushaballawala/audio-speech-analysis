import re, html as html_lib
from datetime import datetime
import pandas as pd, numpy as np

IDX = "/data_store2/resection/neuropsych_video/presidio/Stage2/chronic_pain/RCS08/index.html"
CSV = "/userdata/msharma/chronic_pain_patient_data_1928757PerceptDBSFor_DATA_2026-07-22_1414.csv"

# recording N + datetime from index.html
txt = open(IDX).read()
ROW = re.compile(r"recording_(\d+)_audio_weekly\.m4a\"[^>]*>[^<]*</a></td><td>([^<]+)</td></tr>")
recs = []
for m in ROW.finditer(txt):
    n = m.group(1)
    dt = datetime.strptime(html_lib.unescape(m.group(2)).strip(), "%m/%d/%Y %I:%M%p")
    recs.append((n, pd.Timestamp(dt)))
rec_ts = pd.Series({n: t for n, t in recs}).sort_values()
print(f"recordings parsed from index.html: {len(rec_ts)}")
print(f"recording timespan: {rec_ts.min()} -> {rec_ts.max()}\n")

d = pd.read_csv(CSV, low_memory=False)

DAILY = "stage_1_daily_surveys_vasnrsmpq_timestamp"
AUDIO = "mood_vasaudio_recording_timestamp"
HAMD = "hamilton_selfrating_scale_for_depression_hamd_timestamp"
PVM = "pain_vas_mood_timestamp"
LONG = "stage_0_long_nrsvasmpq_timestamp"
BPI = "brief_pain_inventory_timestamp"
PC = "pain_catastrophizing_scale_timestamp"
NRS = "nrsvas_timestamp"

CANDS = [
    ("mood_vas_s1_weekly (w/audio)", "mood_vas_s1_weekly", AUDIO),
    ("pain_nrs_s1_daily", "pain_nrs_s1_daily", DAILY),
    ("pain_vas_s1_daily", "pain_vas_s1_daily", DAILY),
    ("relief_vas_s1_daily", "relief_vas_s1_daily", DAILY),
    ("mpq_s1_daily", "mpq_s1_daily", DAILY),
    ("left_leg_vas_s1_daily", "left_leg_vas_s1_daily", DAILY),
    ("back_vas_s1_daily", "back_vas_s1_daily", DAILY),
    ("mood_vas_s1_daily", "mood_vas_s1_daily", DAILY),
    ("hamd_score", "hamd_score", HAMD),
    ("hamd_q1", "hamd_q1", HAMD),
    ("mood_vas (pain_vas_mood)", "mood_vas", PVM),
    ("numerical_intensity_2", "numerical_intensity_2", PVM),
    ("numerical_unpleasantness_2", "numerical_unpleasantness_2", PVM),
    ("sf_mpq_sum", "sf_mpq_sum", LONG),
    ("pain_severity_score_bpi", "pain_severity_score_bpi", BPI),
    ("pain_interference_score_bpi", "pain_interference_score_bpi", BPI),
    ("total_score_pc", "total_score_pc", PC),
    ("pain_nrs (nrsvas)", "pain_nrs", NRS),
    ("depression_vas (nrsvas)", "depression_vas", NRS),
    ("anxiety_vas (nrsvas)", "anxiety_vas", NRS),
]

rows = []
for lbl, col, tcol in CANDS:
    if col not in d.columns or tcol not in d.columns:
        rows.append((lbl, col, 0, 0, 0, 0, np.nan)); continue
    sub = d[d[col].notna()].copy()
    sub["ts"] = pd.to_datetime(sub[tcol], errors="coerce")
    sub = sub[sub["ts"].notna()]
    stamps = np.sort(sub["ts"].values)
    if len(stamps) == 0:
        rows.append((lbl, col, 0, 0, 0, 0, np.nan)); continue
    gaps = []
    for t in rec_ts.values:
        i = np.searchsorted(stamps, t)
        cands = []
        if i < len(stamps): cands.append(abs(stamps[i] - t))
        if i > 0: cands.append(abs(t - stamps[i - 1]))
        gaps.append(min(cands) / np.timedelta64(1, "D"))
    gaps = np.array(gaps)
    rows.append((lbl, col, len(stamps), int((gaps <= 1).sum()), int((gaps <= 3).sum()),
                 int((gaps <= 7).sum()), round(gaps.max(), 1)))

R = pd.DataFrame(rows, columns=["score", "column", "n_surveys", "within_1d", "within_3d", "within_7d", "max_gap_days"])
R = R.sort_values(["within_3d", "n_surveys"], ascending=False)
pd.set_option("display.width", 200); pd.set_option("display.max_rows", 100)
print(f"of {len(rec_ts)} recordings, how many match a survey within N days:\n")
print(R.to_string(index=False))
print(f"\n>>> AVAILABLE FOR ALL {len(rec_ts)} RECORDINGS (within 7 days):")
print("   ", R[R.within_7d == len(rec_ts)]["score"].tolist())
R.to_csv("/userdata/msharma/RCS08_jobs/score_coverage.csv", index=False)
