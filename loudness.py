import numpy as np
import parselmouth
from parselmouth.praat import call
import matplotlib.pyplot as plt
import time
import os
import csv


# BE CAREFUL BECAUSE LOUDNESS CAN CHANGE BASED ON MICROPHONE, AND LOUDNESS IS NOT ROBUST TO CONTEXT

snd = "raw_audio/hoarse_test_voice.wav" # mean intensity was 50.64
# snd = "raw_audio/testsoundmono.mp3" # mean intensity was 57.78
# snd = "raw_audio/high_pitch.wav"

def loudness_in_db(
    sound_path: str,
    csv_folder_name: str,
    time_step: float = 0.01,
    pitch_floor: float = 75.0,
    subtract_mean: bool = True,
    activity_threshold_db: float = 40.0, # CAN ADJUST THIS VALUE (NORMALLY AROUND 40)
    stats: bool = True
):
    """
    Mean intensity (dB) over active frames only.
    Returns times (s), intensity_per_frame, mean_intensity
    """
    
    if(stats):
        start_time = time.perf_counter()
        wav_base = os.path.splitext(os.path.basename(sound_path))[0]
        func_name = loudness_in_db.__name__
        stats_csv_file_name = f"{csv_folder_name}/{wav_base}_{func_name}.csv"
    
    sound = parselmouth.Sound(sound_path)
    sampling_hz = sound.sampling_frequency
    
    intensity = call(sound, "To Intensity...", pitch_floor, time_step, subtract_mean) #
    vals = intensity.values[0, :]
    times = intensity.xs()
    
    active_mask = vals >= activity_threshold_db
    active_times = times[active_mask]
    active_intensity_vals = vals[active_mask]
    
    active_intensity_vals_mean = np.mean(active_intensity_vals)
    
    if stats:
        # Timing:
        elapsed_sec = time.perf_counter() - start_time
        with open(stats_csv_file_name, "x", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "sound_path",
                "sample_rate_hz",
                "time_step",
                "pitch_floor",
                "subtract_mean",
                "activity_threshold_db",
                "active_intensity_vals_mean",
                "elapsed_seconds",
            ])
            writer.writerow([
                os.path.basename(sound_path),
                sampling_hz,
                time_step,
                pitch_floor,
                subtract_mean,
                activity_threshold_db,
                active_intensity_vals_mean,
                f"{elapsed_sec:.6f}",
            ])
    
    return active_times, active_intensity_vals, active_intensity_vals_mean, sampling_hz
    
    
    
ts, intensity, mean_intensity, s_hz = loudness_in_db(snd, 'function_output_data')


#__________TESTING___________

# print(mean_intensity)
# plt.scatter(ts, intensity, color='blue', label='Data Points')
# plt.axhline(y=mean_intensity, color='r', linestyle='-', label=f'Intensity mean: {mean_intensity}')
# plt.title("Intensities over time")
# plt.xlabel("Time (in seconds)")
# plt.ylabel("Intensity (dB)")
# plt.savefig("figures/loudnesshoarsevoice.png", dpi=300, bbox_inches='tight')
# plt.show()

    