#!/bin/bash
# RCS08 full audio pipeline: convert+preprocess -> base features -> extra/jitter-shimmer/wpm.
# Runs sequentially where there are dependencies, parallel for the independent extractors.
set -u
PY=/userdata/msharma/myenv/bin/python
A=/userdata/msharma/audio-speech-analysis
LOG=/userdata/msharma/RCS08_jobs; mkdir -p "$LOG"
WD=/data_store2/resection/neuropsych_video/presidio/Stage2/chronic_pain/RCS08/sub-RCS08_stage-1_audio-athome_signal-preproc_spectral_gating_100_percent
PAR=/userdata/msharma/sub-RCS08-stage-1_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots
PRE=sub-RCS08_stage-1_audio-audiotype_preproc_spectral_gating_100_percent

echo "[$(date)] STEP 1: convert + preprocess"
$PY -u "$A/rcs08_convert_preprocess.py" > "$LOG/01_convert_preproc.out" 2>&1
echo "[$(date)] STEP 2: base features (pitch/loudness/f3/alpha)"
$PY -u "$A/rcs08_base_features.py" > "$LOG/02_base_feats.out" 2>&1
echo "[$(date)] STEP 3: extra (hnr/zcr/mfcc/cpps) + jitter/shimmer + wpm in parallel"
$PY -u "$A/extra_features.py"        --wav-dir "$WD" --parent "$PAR" --prefix "$PRE" > "$LOG/03_extra.out" 2>&1 &
$PY -u "$A/robust_jitter_shimmer.py" --wav-dir "$WD" --parent "$PAR" --prefix "$PRE" > "$LOG/04_js.out" 2>&1 &
$PY -u "$A/wpm_extract.py"           --wav-dir "$WD" --parent "$PAR" --prefix "$PRE" --cpu-threads 6 > "$LOG/05_wpm.out" 2>&1 &
wait
echo "[$(date)] RCS08 FEATURE PIPELINE DONE"
touch "$LOG/PIPELINE_DONE"
