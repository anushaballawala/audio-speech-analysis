#!/usr/bin/env python3
"""Words-per-minute (WPM) acoustic feature for PR05 Stage 3.

Transcribes each preprocessed wav with faster-whisper (CPU) and computes
words-per-minute. WPM is words / (total audio duration in minutes); an
articulation-rate variant (words / summed speech-segment time) is also stored.

Sharded via --start/--end so several workers can run in parallel; each writes
one CSV per recording into the wpm_metadata subfolder and skips files already
done. After all shards finish, run with --summary-only to build the summary plot.

Usage:
    python wpm.py --start 1 --end 476 --model base --cpu-threads 6
    python wpm.py --summary-only
"""
import argparse
import csv
import os
import re
import time
from pathlib import Path

import numpy as np

PREPROC_DIR = "/data_store2/resection/neuropsych_video/presidio/Stage3/PR05/sub-PR05_stage-3_audio_signal-preproc_spectral_gating_100_percent"
PARENT = "/userdata/msharma/sub-PR05-stage-3_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots"
WPM_METADATA_DIR = f"{PARENT}/sub-PR05_stage-3_audio-audiotype_preproc_spectral_gating_100_percent_wpm_metadata"
WPM_PLOT_DIR = f"{PARENT}/sub-PR05_stage-3_audio-audiotype_preproc_spectral_gating_100_percent_wpm_plots"

WAV_PREFIX = "sub-PR05_stage-3_audio_signal-preproc_"
SAMPLE_RATE = 16000  # faster-whisper resamples to 16 kHz


def count_words(text: str) -> int:
    # keep apostrophes (contractions count as one word), drop other punctuation
    cleaned = re.sub(r"[^\w\s']", " ", text)
    return len([w for w in cleaned.split() if any(ch.isalnum() for ch in w)])


def transcribe_one(model, audio: np.ndarray):
    segments, _info = model.transcribe(audio, language="en", beam_size=1)
    parts, speech_sec = [], 0.0
    for seg in segments:
        parts.append(seg.text)
        speech_sec += max(0.0, seg.end - seg.start)
    return " ".join(p.strip() for p in parts).strip(), speech_sec


def run_range(start: int, end: int, model_size: str, cpu_threads: int):
    from faster_whisper import WhisperModel, decode_audio

    os.makedirs(WPM_METADATA_DIR, exist_ok=True)
    model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=cpu_threads)

    done = skipped = failed = 0
    for num in range(start, end):
        wav = os.path.join(PREPROC_DIR, f"{WAV_PREFIX}{num}.wav")
        if not os.path.exists(wav):
            continue
        wav_base = os.path.splitext(os.path.basename(wav))[0]
        out_csv = os.path.join(WPM_METADATA_DIR, f"{wav_base}_wpm.csv")
        if os.path.exists(out_csv):
            skipped += 1
            continue
        try:
            t0 = time.perf_counter()
            audio = decode_audio(wav, sampling_rate=SAMPLE_RATE)
            duration_sec = len(audio) / SAMPLE_RATE
            text, speech_sec = transcribe_one(model, audio)
            wc = count_words(text)
            wpm = wc / (duration_sec / 60.0) if duration_sec > 0 else 0.0
            wpm_articulation = wc / (speech_sec / 60.0) if speech_sec > 0 else 0.0
            elapsed = time.perf_counter() - t0
            with open(out_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["sound_path", "model", "sample_rate_hz", "word_count",
                            "duration_sec", "speech_sec", "wpm", "wpm_articulation",
                            "transcript", "elapsed_seconds"])
                w.writerow([wav, model_size, SAMPLE_RATE, wc,
                            f"{duration_sec:.3f}", f"{speech_sec:.3f}",
                            f"{wpm:.4f}", f"{wpm_articulation:.4f}",
                            text, f"{elapsed:.3f}"])
            done += 1
            print(f"[OK]   {num}  words={wc} dur={duration_sec:.1f}s wpm={wpm:.1f}", flush=True)
        except Exception as e:
            failed += 1
            print(f"[FAIL] {num}: {e}", flush=True)
    print(f"\n=== shard {start}-{end-1} DONE === done={done} skipped={skipped} failed={failed}", flush=True)


def build_summary():
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(WPM_PLOT_DIR, exist_ok=True)
    rows = []
    for p in sorted(Path(WPM_METADATA_DIR).glob("*_wpm.csv")):
        m = re.search(r"signal-preproc_(\d+)_", p.name)
        if not m:
            continue
        head = pd.read_csv(p, nrows=1)
        rows.append((int(m.group(1)), float(head.iloc[0]["wpm"])))
    rows.sort()
    labels = [str(r[0]) for r in rows]
    vals = [r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.08), 4))
    ax.scatter(range(len(vals)), vals, s=12, alpha=0.7)
    ax.set_xlabel("Recording")
    ax.set_ylabel("Words per minute")
    ax.set_title("Words per Minute Across Recordings")
    step = max(1, len(labels) // 30)
    ax.set_xticks(range(0, len(labels), step))
    ax.set_xticklabels(labels[::step], rotation=90, fontsize=6)
    fig.tight_layout()
    out = os.path.join(WPM_PLOT_DIR, "sub-PR05_stage-3_wpm_summary.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Summary plot saved to {out}  (n={len(vals)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=476)
    ap.add_argument("--model", default="base")
    ap.add_argument("--cpu-threads", type=int, default=6)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    if args.summary_only:
        build_summary()
    else:
        run_range(args.start, args.end, args.model, args.cpu_threads)


if __name__ == "__main__":
    main()
