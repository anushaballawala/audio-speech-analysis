import os, glob, sys
sys.path.insert(0, "/userdata/msharma/audio-speech-analysis")
from pitch import pitches
from loudness import loudness_in_db
from formants import relative_energy_formant
from alpha_ratio import alpha_ratio

ROOT = "/data_store2/resection/neuropsych_video/presidio/Stage2/chronic_pain/RCS08"
PREPROC = f"{ROOT}/sub-RCS08_stage-1_audio-athome_signal-preproc_spectral_gating_100_percent"
PAR = "/userdata/msharma/sub-RCS08-stage-1_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots"
PRE = "sub-RCS08_stage-1_audio-audiotype_preproc_spectral_gating_100_percent"

PITCH = f"{PAR}/{PRE}_pitch_metadata"
LOUD = f"{PAR}/{PRE}_loudness_metadata"
F3 = f"{PAR}/{PRE}_f3_metadata"
ALPHA = f"{PAR}/{PRE}_alpha_ratio_metadata"
for d in (PITCH, LOUD, F3, ALPHA):
    os.makedirs(d, exist_ok=True)

wavs = sorted(glob.glob(f"{PREPROC}/*.wav"))
print(f"{len(wavs)} preprocessed wavs", flush=True)
ok = 0
for w in wavs:
    for fn, args, name in [(pitches, (w, PITCH), "pitch"),
                           (loudness_in_db, (w, LOUD), "loudness"),
                           (relative_energy_formant, (w, F3, 3), "f3"),
                           (alpha_ratio, (w, ALPHA), "alpha")]:
        try:
            fn(*args)
        except Exception as e:
            print(f"  {name} FAIL {os.path.basename(w)}: {e}", flush=True)
    ok += 1
    if ok % 10 == 0:
        print(f"  base features {ok}/{len(wavs)}...", flush=True)
print(f"DONE base features on {ok} recordings", flush=True)
for d, nm in [(PITCH, "pitch"), (LOUD, "loudness"), (F3, "f3"), (ALPHA, "alpha")]:
    print(f"  {nm}: {len(glob.glob(d+'/*.csv'))} csvs", flush=True)
