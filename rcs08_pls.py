import pandas as pd

# load everything (functions + constants) from pls_regression.py before its __main__
_ns = {}
exec(open("/userdata/msharma/audio-speech-analysis/pls_regression.py").read().split('if __name__ == "__main__"')[0], _ns)
run = _ns["run"]; SUBJECT_RE = _ns["SUBJECT_RE"]; UD = _ns["UD"]

# override the target list (module global that run() reads)
_ns["TARGETS"] = ["pain_nrs_s1_daily", "pain_vas_s1_daily", "relief_vas_s1_daily",
                  "mpq_s1_daily", "mood_vas_s1_weekly"]

PARENT = f"{UD}/sub-RCS08-stage-1_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots"
PREFIX = "sub-RCS08_stage-1_audio-audiotype_preproc_spectral_gating_100_percent"

# scores already matched to each recording in the correlation step
merged = pd.read_csv(f"{PARENT}/{PREFIX}_pain_mood_correlation/merged_scores_features.csv")
score_df = merged[["audio_id"] + _ns["TARGETS"]].rename(columns={"audio_id": "id"})
score_df["id"] = score_df["id"].astype(str)

run("RCS08 Stage 1 (chronic pain)", PARENT, PREFIX, SUBJECT_RE, score_df, PARENT, PREFIX)
