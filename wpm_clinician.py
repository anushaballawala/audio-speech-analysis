#!/usr/bin/env python3
"""Words-per-minute for PR05 ClinicianScales patient-isolated preprocessed audio.

Globs the clinician preproc folder, transcribes each wav with faster-whisper
(base, CPU int8), and writes per-recording WPM CSVs + a summary plot, matching
the clinician-scales folder layout. Idempotent; try/excepts each file.

Usage:
    python wpm_clinician.py
    python wpm_clinician.py --summary-only
"""
import argparse
import csv
import glob
import os
import re
import time
from pathlib import Path

import numpy as np

PREPROC_DIR = "/data_store2/resection/neuropsych_video/presidio/Stage2/ClinicianScales/PR05/PR05_clinician_scales_audio_preproc_spectral_gating_100_percent"
PARENT = "/userdata/msharma/sub-PR05-clinician_scales_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots"
WPM_METADATA_DIR = f"{PARENT}/sub-PR05_clinician_scales_audio-audiotype_preproc_spectral_gating_100_percent_wpm_metadata"
WPM_PLOT_DIR = f"{PARENT}/sub-PR05_clinician_scales_audio-audiotype_preproc_spectral_gating_100_percent_wpm_plots"
SAMPLE_RATE = 16000


def count_words(text):
    cleaned = re.sub(r"[^\w\s']", " ", text)
    return len([w for w in cleaned.split() if any(ch.isalnum() for ch in w)])


def run(model_size="base", cpu_threads=8):
    from faster_whisper import WhisperModel, decode_audio
    os.makedirs(WPM_METADATA_DIR, exist_ok=True)
    model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=cpu_threads)
    done = skipped = failed = 0
    for wav in sorted(glob.glob(f"{PREPROC_DIR}/*.wav")):
        base = os.path.splitext(os.path.basename(wav))[0]
        out_csv = f"{WPM_METADATA_DIR}/{base}_wpm.csv"
        if os.path.exists(out_csv):
            skipped += 1
            continue
        try:
            t0 = time.perf_counter()
            audio = decode_audio(wav, sampling_rate=SAMPLE_RATE)
            dur = len(audio) / SAMPLE_RATE
            segs, _ = model.transcribe(audio, language="en", beam_size=1)
            parts, speech = [], 0.0
            for s in segs:
                parts.append(s.text)
                speech += max(0.0, s.end - s.start)
            text = " ".join(p.strip() for p in parts).strip()
            wc = count_words(text)
            wpm = wc / (dur / 60.0) if dur > 0 else 0.0
            wpm_art = wc / (speech / 60.0) if speech > 0 else 0.0
            with open(out_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["sound_path", "model", "sample_rate_hz", "word_count",
                            "duration_sec", "speech_sec", "wpm", "wpm_articulation",
                            "transcript", "elapsed_seconds"])
                w.writerow([wav, model_size, SAMPLE_RATE, wc, f"{dur:.3f}", f"{speech:.3f}",
                            f"{wpm:.4f}", f"{wpm_art:.4f}", text, f"{time.perf_counter()-t0:.3f}"])
            done += 1
            print(f"[OK] {base}: words={wc} dur={dur:.1f}s wpm={wpm:.1f}", flush=True)
        except Exception as e:
            failed += 1
            print(f"[FAIL] {base}: {e}", flush=True)
    print(f"\n=== WPM DONE === done={done} skipped={skipped} failed={failed}", flush=True)


def build_summary():
    import pandas as pd, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(WPM_PLOT_DIR, exist_ok=True)
    rows = []
    for p in sorted(Path(WPM_METADATA_DIR).glob("*_wpm.csv")):
        d = pd.read_csv(p, nrows=1).iloc[0]
        rows.append((p.name, float(d["wpm"])))
    labels = [r[0].split("_Recording")[0] for r in rows]
    vals = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.35), 4))
    ax.scatter(range(len(vals)), vals, s=20)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("Words per minute")
    ax.set_title("WPM Across Clinician-Scale Recordings")
    fig.tight_layout()
    out = f"{WPM_PLOT_DIR}/sub-PR05_clinician_scales_wpm_summary.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Summary plot saved to {out} (n={len(vals)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    build_summary() if a.summary_only else run()
