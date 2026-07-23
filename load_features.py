"""
gives a dataframe


    df = load_features()                    # every recording, all patients/stages
    df = load_features("PR08 Pre-Stage 2")  # just one dataset

each row is one recording. columns: patient_stage, audio_id, timestamp,
the 59 acoustic features, and the symptom scores (hamd_total, hamd_q1..q6,
vas_anxiety, vas_depression, madrs_total [PR05], vas_lowenergy [PR08/PR09]).
reads the *_all_features.pkl tables built by build_all_features.py
"""
import glob
import pandas as pd

UD = "/userdata/msharma"


def load_features(patient_stage=None):
    pkls = sorted(glob.glob(f"{UD}/sub-*_metadata_and_plots/*_all_features.pkl"))
    df = pd.concat([pd.read_pickle(p) for p in pkls], ignore_index=True)
    if patient_stage is not None:
        df = df[df["patient_stage"] == patient_stage].reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = load_features()
    print("combined dataframe:", df.shape)
    print(df["patient_stage"].value_counts().to_dict())
    df.to_csv(f"{UD}/ALL_patients_features_scores.csv", index=False)
    df.to_pickle(f"{UD}/ALL_patients_features_scores.pkl")
    print("wrote /userdata/msharma/ALL_patients_features_scores.{csv,pkl}")
