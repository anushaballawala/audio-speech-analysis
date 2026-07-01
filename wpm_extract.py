#!/usr/bin/env python3
"""
Parametrized words-per-minute extractor
"""
import argparse, csv, glob, os, re, time
import numpy as np

SAMPLE_RATE = 16000


def count_words(text):
    cleaned = re.sub(r"[^\w\s']", " ", text)
    return len([w for w in cleaned.split() if any(ch.isalnum() for ch in w)])


def run(wav_dir, parent, prefix, model_size, cpu_threads, start, end):
    from faster_whisper import WhisperModel, decode_audio
    md = f"{parent}/{prefix}_wpm_metadata"
    os.makedirs(md, exist_ok=True)
    os.makedirs(f"{parent}/{prefix}_wpm_plots", exist_ok=True)
    model = WhisperModel(model_size, device="cpu", compute_type="int8", cpu_threads=cpu_threads)
    wavs = sorted(glob.glob(f"{wav_dir}/*.wav"))[start:end]
    done = skip = fail = 0
    for wav in wavs:
        base = os.path.splitext(os.path.basename(wav))[0]
        out = f"{md}/{base}_wpm.csv"
        if os.path.exists(out):
            skip += 1; continue
        try:
            t0 = time.perf_counter()
            audio = decode_audio(wav, sampling_rate=SAMPLE_RATE)
            dur = len(audio) / SAMPLE_RATE
            segs, _ = model.transcribe(audio, language="en", beam_size=1)
            parts, speech = [], 0.0
            for s in segs:
                parts.append(s.text); speech += max(0.0, s.end - s.start)
            text = " ".join(p.strip() for p in parts).strip()
            wc = count_words(text)
            wpm = wc / (dur / 60.0) if dur > 0 else 0.0
            wpm_art = wc / (speech / 60.0) if speech > 0 else 0.0
            with open(out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["sound_path","model","sample_rate_hz","word_count","duration_sec",
                            "speech_sec","wpm","wpm_articulation","transcript","elapsed_seconds"])
                w.writerow([wav, model_size, SAMPLE_RATE, wc, f"{dur:.3f}", f"{speech:.3f}",
                            f"{wpm:.4f}", f"{wpm_art:.4f}", text, f"{time.perf_counter()-t0:.3f}"])
            done += 1
            print(f"[OK] {base}: words={wc} dur={dur:.1f}s wpm={wpm:.1f}", flush=True)
        except Exception as e:
            fail += 1; print(f"[FAIL] {base}: {e}", flush=True)
    print(f"=== shard[{start}:{end}] done ok={done} skip={skip} fail={fail} ===", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--parent", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--model", default="base")
    ap.add_argument("--cpu-threads", type=int, default=6)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=10**9)
    a = ap.parse_args()
    run(a.wav_dir, a.parent, a.prefix, a.model, a.cpu_threads, a.start, a.end)
