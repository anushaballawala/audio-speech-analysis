#!/usr/bin/env python3
"""Extract HNR, ZCR, MFCC, CPPS.

Features
  HNR  - Harmonics-to-Noise Ratio (dB), Harmonicity (cc); mean/std/median/iqr
  ZCR  - Zero-Crossing Rate per 25ms frame; mean/std/median/iqr
  MFCC - 13 mel-frequency cepstral coefficients (C0..C12), per-coeff frame mean
  CPPS - Smoothed Cepstral Peak Prominence (dB), single recording-level value

"""
import argparse, csv, glob, os
import numpy as np
import parselmouth
from parselmouth.praat import call

FEATS = ["hnr", "zcr", "mfcc", "cpps"]


def hnr_stats(snd):
    h = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    v = h.values.ravel()                   # already a float64 ndarray
    v = v[np.isfinite(v) & (v != -200)]    # drop NaN/inf AND Praat's unvoiced (-200) sentinel
    if v.size == 0:
        return [np.nan]*6
    q1, q3 = np.percentile(v, [25, 75])
    return [float(np.mean(v)), float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
            float(np.median(v)), float(q1), float(q3), float(q3 - q1)]


def zcr_stats(snd, frame_s=0.025, hop_s=0.010):
    x = np.asarray(snd.values[0], dtype=float)
    sr = snd.sampling_frequency
    n = max(1, int(frame_s * sr)); hop = max(1, int(hop_s * sr))
    vals = []
    for s in range(0, max(1, len(x) - n + 1), hop):
        fr = x[s:s+n]
        vals.append(np.mean(np.abs(np.diff(np.sign(fr))) > 0))
    v = np.array(vals)
    v = v[np.isfinite(v)]                   # future-proof: drop any NaN/inf frame
    if v.size == 0:
        return [np.nan]*6
    q1, q3 = np.percentile(v, [25, 75])
    return [float(np.mean(v)), float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
            float(np.median(v)), float(q1), float(q3), float(q3 - q1)]


def mfcc_means(snd, n=12):
    mf = snd.to_mfcc(number_of_coefficients=n)        # yields C0..Cn (n+1 rows)
    arr = np.asarray(mf.to_array())
    return [float(np.mean(arr[i])) for i in range(arr.shape[0])]   # 13 means


def cpps_value(snd):
    pc = call(snd, "To PowerCepstrogram", 60, 0.002, 5000, 50)
    return float(call(pc, "Get CPPS", "yes", 0.02, 0.0005, 60, 330, 0.05,
                      "parabolic", 0.001, 0, "Straight", "Robust"))


def process(wav, parent, prefix):
    base = os.path.splitext(os.path.basename(wav))[0]
    outs = {f: f"{parent}/{prefix}_{f}_metadata/{base}_{f}.csv" for f in FEATS}
    if all(os.path.exists(p) for p in outs.values()):
        return "skip"
    snd = parselmouth.Sound(wav)
    sr = snd.sampling_frequency

    hnr = hnr_stats(snd)
    with open(outs["hnr"], "w", newline="") as f:
        w = csv.writer(f); w.writerow(["sound_path","sample_rate_hz","hnr_mean","hnr_std","hnr_median","hnr_q1","hnr_q3","hnr_iqr"]); w.writerow([base, sr, *hnr])

    zcr = zcr_stats(snd)
    with open(outs["zcr"], "w", newline="") as f:
        w = csv.writer(f); w.writerow(["sound_path","sample_rate_hz","zcr_mean","zcr_std","zcr_median","zcr_q1","zcr_q3","zcr_iqr"]); w.writerow([base, sr, *zcr])

    mf = mfcc_means(snd)
    with open(outs["mfcc"], "w", newline="") as f:
        w = csv.writer(f); cols = [f"mfcc_c{i}_mean" for i in range(len(mf))]; w.writerow(["sound_path","sample_rate_hz",*cols]); w.writerow([base, sr, *mf])

    cpps = cpps_value(snd)
    with open(outs["cpps"], "w", newline="") as f:
        w = csv.writer(f); w.writerow(["sound_path","sample_rate_hz","cpps"]); w.writerow([base, sr, cpps])
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--parent", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--glob", default="*.wav")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=10**9)
    a = ap.parse_args()
    for f in FEATS:
        os.makedirs(f"{a.parent}/{a.prefix}_{f}_metadata", exist_ok=True)
        os.makedirs(f"{a.parent}/{a.prefix}_{f}_plots", exist_ok=True)
    wavs = sorted(glob.glob(os.path.join(a.wav_dir, a.glob)))[a.start:a.end]
    ok = sk = fail = 0
    for w in wavs:
        try:
            r = process(w, a.parent, a.prefix)
            if r == "ok": ok += 1; print(f"[OK] {os.path.basename(w)}", flush=True)
            else: sk += 1
        except Exception as e:
            fail += 1; print(f"[FAIL] {os.path.basename(w)}: {e}", flush=True)
    print(f"=== shard[{a.start}:{a.end}] done ok={ok} skip={sk} fail={fail} ===", flush=True)


if __name__ == "__main__":
    main()
