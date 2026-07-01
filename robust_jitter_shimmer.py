#!/usr/bin/env python3
"""Robust jitter + shimmer extraction
Usage:
  python robust_jitter_shimmer.py --wav-dir DIR --parent PARENT --prefix PREFIX
"""

import argparse, csv, glob, os, math, time
import parselmouth
from parselmouth.praat import call

JITTER_FLOORS = [115.0, 90.0, 75.0, 60.0, 50.0]   # first = original jitter.py floor
JITTER_CEIL = 400.0
SHIMMER_FLOORS = [75.0, 60.0, 50.0]               # first = original shimmer.py floor
SHIMMER_CEIL = 500.0
TIME_STEP = 0.01
PERIOD_FLOOR, PERIOD_CEIL, MAX_PERIOD_FACTOR = 0.0001, 0.02, 1.3
MAX_AMP_FACTOR = 1.6


def _valid(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def jitter_with_fallback(snd):
    for floor in JITTER_FLOORS:
        try:
            pitch = snd.to_pitch(time_step=TIME_STEP, pitch_floor=floor, pitch_ceiling=JITTER_CEIL)
            point = call([snd, pitch], "To PointProcess (cc)")
            jv = call(point, "Get jitter (local)", 0, 0, PERIOD_FLOOR, PERIOD_CEIL, MAX_PERIOD_FACTOR)
            if _valid(jv):
                return jv, floor
        except Exception:
            continue
    return float("nan"), JITTER_FLOORS[-1]


def shimmer_with_fallback(snd, N=5):
    for floor in SHIMMER_FLOORS:
        try:
            pitch = snd.to_pitch(time_step=TIME_STEP, pitch_floor=floor, pitch_ceiling=SHIMMER_CEIL)
            point = call([snd, pitch], "To PointProcess (cc)")
            sv = call([snd, point], f"Get shimmer (apq{N})", 0, 0, PERIOD_FLOOR, PERIOD_CEIL,
                      MAX_PERIOD_FACTOR, MAX_AMP_FACTOR)
            if _valid(sv):
                return sv, floor
        except Exception:
            continue
    return float("nan"), SHIMMER_FLOORS[-1]


def existing_valid(path, col):
    if not os.path.exists(path):
        return False
    try:
        import pandas as pd
        v = pd.read_csv(path, nrows=1).iloc[0].get(col)
        return _valid(float(v))
    except Exception:
        return False


def run(wav_dir, parent, prefix):
    jmd = f"{parent}/{prefix}_jitter_metadata"
    smd = f"{parent}/{prefix}_shimmer_metadata"
    for d in (jmd, smd, f"{parent}/{prefix}_jitter_plots", f"{parent}/{prefix}_shimmer_plots"):
        os.makedirs(d, exist_ok=True)
    jfix = sfix = jskip = sskip = jfail = sfail = 0
    for wav in sorted(glob.glob(f"{wav_dir}/*.wav")):
        base = os.path.splitext(os.path.basename(wav))[0]
        jout, sout = f"{jmd}/{base}_jitter.csv", f"{smd}/{base}_shimmer_apqN.csv"
        need_j = not existing_valid(jout, "jitter_val")
        need_s = not existing_valid(sout, "shimmer_val")
        if not need_j and not need_s:
            jskip += 1; sskip += 1; continue
        snd = parselmouth.Sound(wav); sr = snd.sampling_frequency
        if need_j:
            t0 = time.perf_counter(); jv, floor = jitter_with_fallback(snd)
            with open(jout, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["sound_path","sample_rate_hz","kind","pitch_floor","pitch_ceiling","pitch_time_step",
                            "from_time","to_time","period_floor","period_ceiling","maximum_period_factor",
                            "jitter_val","elapsed_seconds"])
                w.writerow([os.path.basename(wav), sr, "local", floor, JITTER_CEIL, TIME_STEP, 0, 0,
                            PERIOD_FLOOR, PERIOD_CEIL, MAX_PERIOD_FACTOR, jv, f"{time.perf_counter()-t0:.6f}"])
            jfix += 1; jfail += int(not _valid(jv))
            print(f"[J {'ok' if _valid(jv) else 'FAIL'}] {base} floor={floor} jitter={jv}", flush=True)
        if need_s:
            t0 = time.perf_counter(); sv, floor = shimmer_with_fallback(snd, 5)
            with open(sout, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["sound_path","sample_rate_hz","N","pitch_floor","pitch_ceiling","pitch_time_step",
                            "from_time","to_time","period_floor","period_ceiling","maximum_period_factor",
                            "maximum_amplitude_factor","shimmer_val","elapsed_seconds"])
                w.writerow([os.path.basename(wav), sr, 5, floor, SHIMMER_CEIL, TIME_STEP, 0, 0,
                            PERIOD_FLOOR, PERIOD_CEIL, MAX_PERIOD_FACTOR, MAX_AMP_FACTOR, sv, f"{time.perf_counter()-t0:.6f}"])
            sfix += 1; sfail += int(not _valid(sv))
            print(f"[S {'ok' if _valid(sv) else 'FAIL'}] {base} floor={floor} shimmer={sv}", flush=True)
    print(f"=== done jitter(written={jfix} skip={jskip} still_nan={jfail}) "
          f"shimmer(written={sfix} skip={sskip} still_nan={sfail}) ===", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--parent", required=True)
    ap.add_argument("--prefix", required=True)
    a = ap.parse_args()
    run(a.wav_dir, a.parent, a.prefix)
