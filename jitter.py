import numpy as np
import parselmouth
from parselmouth.praat import call
import matplotlib.pyplot as plt
import time
import os
import csv

# snd = "raw_audio/hoarse_test_voice.wav" # jitter value: 0.07892894876979728 (higher, as expected)
snd = "raw_audio/testsoundmono.mp3" # jitter value: 0.02721768951093052
# snd = "raw_audio/high_pitch.wav"


def jitter(
    sound_path: str,
    csv_folder_name: str,
    kind : str = "local",
    pitch_floor: float = 75.0, #currently set to 75Hz, seems to work well
    pitch_ceiling: float = 500.0,
    pitch_time_step = 0.01,
    from_time: float = 0.0, # if from time and to time are same it goes for the entire audio recording
    to_time: float = 0.0,
    period_floor: float = 0.0001, # this and below are default praat vals
    period_ceiling: float = 0.02,
    maximum_period_factor: float = 1.3,
    stats: bool = True
):
    """
    Returns jitter of sound.
    
    kind options: "local", "local, absolute", "rap", "ppq5", "ddp"
    """
    
    if(stats):
        start_time = time.perf_counter()
        wav_base = os.path.splitext(os.path.basename(sound_path))[0]
        func_name = jitter.__name__
        stats_csv_file_name = f"{csv_folder_name}/{wav_base}_{func_name}.csv"
    
    sound = parselmouth.Sound(sound_path)
    sampling_hz = sound.sampling_frequency

    if kind not in ["local", "local, absolute", "rap", "ppq5", "ddp"]:
        raise ValueError("Kind option not one of those allowed. Look at docstring for kind options.")
    
    pitch = sound.to_pitch(time_step=pitch_time_step, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    
    point = call([sound, pitch], "To PointProcess (cc)")

    jtr = call(
        point,
        f"Get jitter ({kind})",
        from_time,
        to_time,
        period_floor,
        period_ceiling,
        maximum_period_factor,
    )
    
    if stats:
        # Timing:
        elapsed_sec = time.perf_counter() - start_time
        with open(stats_csv_file_name, "x", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "sound_path",
                "sample_rate_hz",
                "kind",
                "pitch_floor",
                "pitch_ceiling",
                "pitch_time_step",
                "from_time",
                "to_time",
                "period_floor",
                "period_ceiling",
                "maximum_period_factor",
                "jitter_val",
                "elapsed_seconds",
            ])
            writer.writerow([
                os.path.basename(sound_path),
                sampling_hz,
                kind,
                pitch_floor,
                pitch_ceiling,
                pitch_time_step,
                from_time,
                to_time,
                period_floor,
                period_ceiling,
                maximum_period_factor,
                jtr,
                f"{elapsed_sec:.6f}",
            ])
    
    return jtr, sampling_hz

jtr_val, s_hz = jitter(snd, 'function_output_data')

print(jtr_val)