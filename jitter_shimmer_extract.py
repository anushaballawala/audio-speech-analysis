#!/usr/bin/env python3
"""Parametrized jitter + shimmer extractor.

Reuses the pipeline's existing jitter() and shimmer_apqN() functions (same Praat
calls + CSV columns as every prior run) over a globbed wav dir, writing into
<parent>/<prefix>_{jitter,shimmer}_metadata. Idempotent (skips existing); each
file try/excepted.

Usage:
  python jitter_shimmer_extract.py --wav-dir DIR --parent PARENT --prefix PREFIX
"""
import argparse, glob, os, sys
sys.path.insert(0, "/userdata/msharma/audio-speech-analysis")
from jitter import jitter
from shimmer import shimmer_apqN


def run(wav_dir, parent, prefix):
    jmd = f"{parent}/{prefix}_jitter_metadata"
    smd = f"{parent}/{prefix}_shimmer_metadata"
    for d in (jmd, smd, f"{parent}/{prefix}_jitter_plots", f"{parent}/{prefix}_shimmer_plots"):
        os.makedirs(d, exist_ok=True)
    done = skip = fail = 0
    for wav in sorted(glob.glob(f"{wav_dir}/*.wav")):
        base = os.path.splitext(os.path.basename(wav))[0]
        jout = f"{jmd}/{base}_jitter.csv"
        sout = f"{smd}/{base}_shimmer_apqN.csv"
        if os.path.exists(jout) and os.path.exists(sout):
            skip += 1; continue
        try:
            if not os.path.exists(jout):
                jitter(wav, jmd)                 # writes {base}_jitter.csv (jitter_val)
            if not os.path.exists(sout):
                shimmer_apqN(wav, smd, 5)        # writes {base}_shimmer_apqN.csv (shimmer_val)
            done += 1
            print(f"[OK] {base}", flush=True)
        except Exception as e:
            fail += 1; print(f"[FAIL] {base}: {e}", flush=True)
    print(f"=== jitter/shimmer done ok={done} skip={skip} fail={fail} ===", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--parent", required=True)
    ap.add_argument("--prefix", required=True)
    a = ap.parse_args()
    run(a.wav_dir, a.parent, a.prefix)
