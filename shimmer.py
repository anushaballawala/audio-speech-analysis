# apqN shimmer is N-point period amplitude perturbation quotient (apq5 is supposed to be most effective)

import numpy as np
import parselmouth
from parselmouth.praat import call
import matplotlib.pyplot as plt
import time
import os
import csv

snd = "raw_audio/hoarse_test_voice.wav" # apq5 shimmer value: 0.11146423422694232 (higher, as expected)
# snd = "raw_audio/testsoundmono.mp3" # apq5 shimmer value: 0.05805759435795879
# snd = "raw_audio/high_pitch.wav"

def shimmer_apqN(
    sound_path: str,
    csv_folder_name: str,
    N: int, # N can only be 3, 5, or 11
    *,
    pitch_floor: float = 75.0, 
    pitch_ceiling: float = 500.0,
    pitch_time_step = 0.01,
    from_time: float = 0.0, # if "from time" and "to time" are same it goes for the entire audio recording
    to_time: float = 0.0,
    period_floor: float = 0.0001, # all of these are default praat values
    period_ceiling: float = 0.02,
    maximum_period_factor: float = 1.3,
    maximum_amplitude_factor: float = 1.6,
    stats: bool = True
) -> float:
    """
    Compute N-point Amplitude Perturbation Quotient (aqpN shimmer)
    """
    
    sound = parselmouth.Sound(sound_path)
    sampling_hz = sound.sampling_frequency
    
    if(stats):
        start_time = time.perf_counter()
        wav_base = os.path.splitext(os.path.basename(sound_path))[0]
        func_name = shimmer_apqN.__name__
        stats_csv_file_name = f"{csv_folder_name}/{wav_base}_{func_name}.csv"
    
    if N not in [3, 5, 11]:
        raise ValueError("N can only be 3, 5, or 11")
    
    pitch = sound.to_pitch(time_step=pitch_time_step, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    
    point = call([sound, pitch], "To PointProcess (cc)")

    apqN = call(
        [sound, point],
        f"Get shimmer (apq{N})",
        from_time,
        to_time,
        period_floor,
        period_ceiling,
        maximum_period_factor,
        maximum_amplitude_factor,
    )
    
    
    if stats:
        # Timing:
        elapsed_sec = time.perf_counter() - start_time
        with open(stats_csv_file_name, "x", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "sound_path",
                "sample_rate_hz",
                "pitch_floor",
                "pitch_ceiling",
                "pitch_time_step",
                "from_time",
                "to_time",
                "period_floor",
                "period_ceiling",
                "maximum_period_factor",
                "maximum_amplitude_factor",
                "shimmer_val"
                "elapsed_seconds"
            ])
            writer.writerow([
                os.path.basename(sound_path),
                sampling_hz,
                pitch_floor,
                pitch_ceiling,
                pitch_time_step,
                from_time,
                to_time,
                period_floor,
                period_ceiling,
                maximum_period_factor,
                maximum_amplitude_factor,
                apqN,
                f"{elapsed_sec:.6f}",
            ])
            
    return apqN, sampling_hz

apq5_shimmer, s_hz = shimmer_apqN(snd, 'function_output_data', 5)
print(apq5_shimmer)
    