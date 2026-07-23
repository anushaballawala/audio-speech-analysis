import glob, os, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VTT_DIR = "/data_store2/resection/neuropsych_video/presidio/Stage2/ClinicianScales/PR05/Transcripts"
OUT = "/userdata/msharma/clinician_overlap_analysis"
TS = re.compile(r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})")
PATIENT = {"vivian", "naiviv's iphone"}   # everyone else = clinician/staff


def norm(name):
    return name.strip().lower().replace("’", "'").replace("‘", "'")


def parse_cues(path):
    lines = open(path, errors="ignore").read().splitlines()
    cues, i, last = [], 0, None
    while i < len(lines):
        if "-->" in lines[i]:
            m = TS.findall(lines[i])
            if len(m) >= 2:
                s = int(m[0][0])*3600 + int(m[0][1])*60 + int(m[0][2]) + int(m[0][3])/1000
                e = int(m[1][0])*3600 + int(m[1][1])*60 + int(m[1][2]) + int(m[1][3])/1000
                j, spk = i+1, None
                while j < len(lines) and lines[j].strip():
                    if ":" in lines[j] and spk is None:
                        spk = norm(lines[j].split(":", 1)[0])
                    j += 1
                spk = spk or last
                if spk and e > s:
                    cues.append((s, e, spk)); last = spk
                i = j; continue
        i += 1
    return cues


def merge(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def intersect(A, B):
    i = j = 0; out = []
    while i < len(A) and j < len(B):
        s, e = max(A[i][0], B[j][0]), min(A[i][1], B[j][1])
        if s < e:
            out.append((s, e))
        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1
    return out


rows, seg_durs = [], []
for f in sorted(glob.glob(f"{VTT_DIR}/*.vtt")):
    cues = parse_cues(f)
    if not cues:
        continue
    pat = merge([(s, e) for s, e, sp in cues if sp in PATIENT])
    clin = merge([(s, e) for s, e, sp in cues if sp not in PATIENT])
    ov = intersect(pat, clin)
    span = max(e for _, e, _ in cues) - min(s for s, _, _ in cues)
    pat_sec = sum(e - s for s, e in pat)
    ov_sec = sum(e - s for s, e in ov)
    seg_durs += [e - s for s, e in ov]
    rows.append({"recording": os.path.basename(f).split(".")[0],
                 "overlap_instances": len(ov), "overlap_seconds": round(ov_sec, 2),
                 "patient_speech_sec": round(pat_sec, 2),
                 "overlap_pct_of_patient_speech": round(100*ov_sec/pat_sec, 2) if pat_sec else 0,
                 "recording_span_sec": round(span, 1)})

df = pd.DataFrame(rows).sort_values("overlap_seconds", ascending=False)
os.makedirs(OUT, exist_ok=True)
df.to_csv(f"{OUT}/overlap_per_recording.csv", index=False)

print(f"recordings analyzed: {len(df)}")
print(f"overlap INSTANCES per recording:  median={df.overlap_instances.median():.0f}  mean={df.overlap_instances.mean():.1f}  max={df.overlap_instances.max()}")
print(f"overlap SECONDS per recording:    median={df.overlap_seconds.median():.1f}  mean={df.overlap_seconds.mean():.1f}  max={df.overlap_seconds.max()}")
print(f"overlap as % of patient speech:   median={df.overlap_pct_of_patient_speech.median():.1f}%  mean={df.overlap_pct_of_patient_speech.mean():.1f}%  max={df.overlap_pct_of_patient_speech.max()}%")
print(f"total individual overlap segments across all recordings: {len(seg_durs)}")

fig, ax = plt.subplots(2, 2, figsize=(13, 9))
ax[0, 0].hist(df.overlap_instances, bins=15, color="#C44E52", edgecolor="k")
ax[0, 0].set_title("Overlap INSTANCES per recording"); ax[0, 0].set_xlabel("# simultaneous-speech events"); ax[0, 0].set_ylabel("# recordings")
ax[0, 1].hist(df.overlap_seconds, bins=15, color="#4C72B0", edgecolor="k")
ax[0, 1].set_title("Overlap SECONDS per recording"); ax[0, 1].set_xlabel("total simultaneous-speech time (s)"); ax[0, 1].set_ylabel("# recordings")
ax[1, 0].hist(df.overlap_pct_of_patient_speech, bins=15, color="#55A868", edgecolor="k")
ax[1, 0].set_title("Overlap as % of the patient's speech time"); ax[1, 0].set_xlabel("overlap seconds / patient speech seconds (%)"); ax[1, 0].set_ylabel("# recordings")
ax[1, 1].hist(seg_durs, bins=30, color="#8172B3", edgecolor="k")
ax[1, 1].set_title(f"Duration of individual overlap events (all {len(seg_durs)} pooled)"); ax[1, 1].set_xlabel("overlap event length (s)"); ax[1, 1].set_ylabel("# events")
fig.suptitle(f"PR05 Clinician Scales: simultaneous-speech (patient + clinician) — {len(df)} recordings", fontsize=14)
fig.tight_layout()
fig.savefig(f"{OUT}/overlap_histograms.png", dpi=150)
plt.close(fig)
print(f"\nwrote {OUT}/overlap_histograms.png + overlap_per_recording.csv")
