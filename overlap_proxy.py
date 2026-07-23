import glob, os, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VTT_DIR = "/data_store2/resection/neuropsych_video/presidio/Stage2/ClinicianScales/PR05/Transcripts"
OUT = "/userdata/msharma/clinician_overlap_analysis"
TS = re.compile(r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})")
PATIENT = {"vivian", "naiviv's iphone"}
TIGHT = 0.05   # seconds; boundary gap <= this = crosstalk-risk transition


def norm(n):
    return n.strip().lower().replace("’", "'").replace("‘", "'")


def parse_cues(path):
    L = open(path, errors="ignore").read().splitlines()
    cues, i, last = [], 0, None
    while i < len(L):
        if "-->" in L[i]:
            m = TS.findall(L[i])
            if len(m) >= 2:
                s = int(m[0][0])*3600+int(m[0][1])*60+int(m[0][2])+int(m[0][3])/1000
                e = int(m[1][0])*3600+int(m[1][1])*60+int(m[1][2])+int(m[1][3])/1000
                j, spk = i+1, None
                while j < len(L) and L[j].strip():
                    if ":" in L[j] and spk is None:
                        spk = norm(L[j].split(":", 1)[0])
                    j += 1
                spk = spk or last
                if spk and e > s:
                    cues.append((s, e, "patient" if spk in PATIENT else "clinician")); last = spk
                i = j; continue
        i += 1
    return cues


rows, all_gaps = [], []
for f in sorted(glob.glob(f"{VTT_DIR}/*.vtt")):
    c = parse_cues(f)
    n_bound = n_tight = 0
    for a, b in zip(c, c[1:]):
        if a[2] != b[2]:                      # patient<->clinician switch
            gap = b[0] - a[1]
            n_bound += 1
            all_gaps.append(gap)
            if gap <= TIGHT:
                n_tight += 1
    n_turns = len(c)
    rows.append({"recording": os.path.basename(f).split(".")[0],
                 "pat_clin_boundaries": n_bound,
                 "tight_boundaries": n_tight,
                 "pct_tight": round(100*n_tight/n_bound, 1) if n_bound else 0,
                 "n_turns": n_turns})

df = pd.DataFrame(rows).sort_values("tight_boundaries", ascending=False)
os.makedirs(OUT, exist_ok=True)
df.to_csv(f"{OUT}/crosstalk_proxy_per_recording.csv", index=False)

print(f"recordings: {len(df)}")
print(f"tight (<= {TIGHT}s) patient<->clinician boundaries per recording:  "
      f"median={df.tight_boundaries.median():.0f}  mean={df.tight_boundaries.mean():.1f}  max={df.tight_boundaries.max()}")
print(f"as % of all patient<->clinician switches:  median={df.pct_tight.median():.0f}%  mean={df.pct_tight.mean():.0f}%")
print(f"total patient<->clinician boundaries pooled: {len(all_gaps)}  |  at exactly 0s: {sum(g<=0.001 for g in all_gaps)}")

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
ax[0].hist(df.tight_boundaries, bins=15, color="#C44E52", edgecolor="k")
ax[0].set_title(f"Tight patient↔clinician boundaries (gap ≤ {TIGHT}s)\nper recording")
ax[0].set_xlabel("# crosstalk-risk transitions"); ax[0].set_ylabel("# recordings")
ax[1].hist(df.pct_tight, bins=15, color="#DD8452", edgecolor="k")
ax[1].set_title("Share of speaker switches that are tight\nper recording")
ax[1].set_xlabel("% of patient↔clinician switches with gap ≤ 0.05s"); ax[1].set_ylabel("# recordings")
gz = [g for g in all_gaps if 0 <= g <= 2]
ax[2].hist(gz, bins=40, color="#4C72B0", edgecolor="k")
ax[2].axvline(TIGHT, color="red", ls="--", lw=1, label=f"tight cutoff {TIGHT}s")
ax[2].set_title(f"Gap between adjacent patient/clinician turns\n(pooled, 0–2s; {len(all_gaps)} boundaries total)")
ax[2].set_xlabel("gap (s)"); ax[2].set_ylabel("# boundaries"); ax[2].legend()
fig.suptitle("PR05 Clinician Scales: crosstalk-risk PROXY from Zoom transcript "
             "(turn-boundary tightness — NOT true overlap seconds)", fontsize=13)
fig.tight_layout()
fig.savefig(f"{OUT}/crosstalk_proxy_histograms.png", dpi=150)
plt.close(fig)
print(f"\nwrote {OUT}/crosstalk_proxy_histograms.png + crosstalk_proxy_per_recording.csv")
