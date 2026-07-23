import os, re, glob, subprocess, sys
sys.path.insert(0, "/userdata/msharma/audio-speech-analysis")
from preprocessing import demean_butterworth_and_denoise

ROOT = "/data_store2/resection/neuropsych_video/presidio/Stage2/chronic_pain/RCS08"
SRC = f"{ROOT}/documents"
RAW = f"{ROOT}/RCS08_raw_audio_wav"
PREPROC = f"{ROOT}/sub-RCS08_stage-1_audio-athome_signal-preproc_spectral_gating_100_percent"
PP_PREFIX = "sub-RCS08_stage-1_audio-athome_signal-preproc"
PAR = "/userdata/msharma/sub-RCS08-stage-1_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots"
PRE = "sub-RCS08_stage-1_audio-audiotype_preproc_spectral_gating_100_percent"
META = f"{PAR}/{PRE}_metadata"
for d in (RAW, PREPROC, META):
    os.makedirs(d, exist_ok=True)

NUM = re.compile(r"recording_(\d+)_audio_weekly")
m4as = sorted(glob.glob(f"{SRC}/*.m4a"))
print(f"{len(m4as)} m4a files found", flush=True)
conv = skip = pp = fail = 0
for f in m4as:
    mm = NUM.search(os.path.basename(f))
    if not mm:
        print("  NO ID:", os.path.basename(f)); fail += 1; continue
    n = mm.group(1)
    raw = f"{RAW}/{n}_audio.wav"
    ppwav = f"{PREPROC}/{PP_PREFIX}_{n}.wav"
    if not os.path.exists(raw):
        r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", f,
                            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", raw])
        if r.returncode != 0:
            print("  FFMPEG FAIL", n); fail += 1; continue
        conv += 1
    if os.path.exists(ppwav):
        skip += 1; continue
    try:
        demean_butterworth_and_denoise(raw, ppwav, META)
        pp += 1
        if pp % 10 == 0:
            print(f"  preprocessed {pp}...", flush=True)
    except Exception as e:
        print(f"  PREPROC FAIL {n}: {e}"); fail += 1
print(f"DONE convert+preprocess: converted={conv} preprocessed={pp} skipped={skip} failed={fail}", flush=True)
print("preproc wavs now:", len(glob.glob(f"{PREPROC}/*.wav")), flush=True)
