import parselmouth
import numpy as np
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
import time
import os
import csv

# snd = "raw_audio/hoarse_test_voice.wav" # pitch mean: 287.3200488390224 (couldn't reliably find pitch though)
snd = "raw_audio/testsoundmono.mp3" # pitch mean: 116



def pitches(
    sound_path: str,
    csv_folder_name: str,
    time_step: float = 0.01,
    pitch_floor: float = 75.0,
    pitch_ceiling: float = 500.0,
    stats: bool = True
):
    """
    Returns:
      times: np.ndarray (s)
      f0_values: np.ndarray
      f0_lstsq_slope: float
      f0_lstsq_intercept: float
    """
    
    sound = parselmouth.Sound(sound_path)
    sampling_hz = sound.sampling_frequency
    #uses praat's autocorrelation method (instead of cc [cross correlation])
    
    if(stats):
        start_time = time.perf_counter()
        wav_base = os.path.splitext(os.path.basename(sound_path))[0]
        func_name = pitches.__name__
        stats_csv_file_name = f"{csv_folder_name}/{wav_base}_{func_name}.csv"
    
    pitch = sound.to_pitch(time_step=time_step, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling) 
    # Extract the frequencies (Hz) and corresponding times
    f0_values = pitch.selected_array["frequency"]  # (in Hz)
    times = pitch.xs()                             # time pts (s)
    nonzero_mask = f0_values != 0
    nonzero_f0_values = f0_values[nonzero_mask]
    nonzero_xs = times[nonzero_mask]

    # least squares Ax = b
    A = np.column_stack((nonzero_xs, np.ones(len(nonzero_xs))))
    b = nonzero_f0_values

    lstsqsoln = lstsq(A, b)[0]

    f0_lstsq_slope = lstsqsoln[0] #measure referenced in paper as a strong feature
    f0_lstsq_intercept = lstsqsoln[1]
    
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
                "pitch_ceiling",
                "f0_lstsq_slope",
                "pitch_mean",
                "elapsed_seconds",
            ])
            writer.writerow([
                os.path.basename(sound_path),
                sampling_hz,
                time_step,
                pitch_floor,
                pitch_ceiling,
                f0_lstsq_slope,
                np.mean(nonzero_f0_values),
                f"{elapsed_sec:.6f}",
            ])
    
    return nonzero_xs, nonzero_f0_values, f0_lstsq_slope, f0_lstsq_intercept, sampling_hz


nonzero_xs, nonzero_f0_values, f0_lstsq_slope, f0_lstsq_intercept, s_hz = pitches(snd, 'function_output_data')


#__________TESTING___________
# print("Pitch mean:", np.mean(nonzero_f0_values))
# line_y_values = f0_lstsq_slope * nonzero_xs + f0_lstsq_intercept
# plt.scatter(nonzero_xs, nonzero_f0_values, color='blue', label='Data Points')
# plt.plot(nonzero_xs, line_y_values, color='red', linestyle='-', label='Line of Best Fit')
# plt.title("Nonzero f0 values")
# plt.xlabel("Time (in seconds)")
# plt.ylabel("Hertz")
# # plt.savefig("figures/hoarse_test_sound_mono_100floor_f0_and_lineoffit.png", dpi=300, bbox_inches='tight')
# plt.show()
