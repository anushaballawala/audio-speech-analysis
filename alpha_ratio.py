import parselmouth
import numpy as np
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
import time
import os
import csv

# # snd = "raw_audio/hoarse_test_voice.wav"
# snd = "raw_audio/testsoundmono.mp3" # gives mean alpha ratio 2.8307666078269107 w/ log10
snd = "raw_audio/high_pitch.wav" # gives mean alpha ratio of 1.4452959901226703 w/ log10


def alpha_ratio( #TODO ask if I should add masking to alpha_ratio
    sound_path: str,
    csv_folder_name: str,
    time_step: float = 0.01, 
    window_length: float = 0.025,
    f_low1: float = 50.0,
    f_high1: float = 1000.0,
    f_low2: float = 1000.0,
    f_high2: float = 5000.0,
    pitch_floor: float = 75.0,
    pitch_ceiling: float = 500.0,
    stats: bool = True
):
    """
    Linear alpha ratio per frame:
         alpha(t) = log10(sum(E[50-1000 Hz]) / sum(E[1000-5000 Hz]))
         
    Args:
    sound_path: path to .csv sound file
    csv_folder_name (str): name of folder where stats csv file should be placed (doesn't matter if stats is False).
    stats (bool): Whether the function should output a csv including time taken to execute and other metadata.
    
     Returns: speaking_times (s), alpha_ratio_per_frame, alpha_ratio_mean
    """
    if(stats):
        start_time = time.perf_counter()
        wav_base = os.path.splitext(os.path.basename(sound_path))[0]
        func_name = alpha_ratio.__name__
        stats_csv_file_name = f"{csv_folder_name}/{wav_base}_{func_name}.csv"
    
    #NEW: FOR MASKING USING F0 (PITCH):
    # pitch = sound.to_pitch(time_step=time_step, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    
    sound = parselmouth.Sound(sound_path)
    sampling_hz = sound.sampling_frequency

    pitch = sound.to_pitch(time_step=time_step, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    
    spectrogram = sound.to_spectrogram(
         window_length=window_length,
         time_step=time_step,
         maximum_frequency=f_high2
    )
    
    frequency_PSDs = spectrogram.values # power spectrum densities (PSD)
    freqs = spectrogram.ys()
    times = spectrogram.xs()

    
    low_freq_mask = (freqs >= f_low1) & (freqs < f_high1)
    high_freq_mask = (freqs >= f_low2) & (freqs <= f_high2)
    
    low_freq_PSDs = frequency_PSDs[low_freq_mask, :]
    high_freq_PSDs = frequency_PSDs[high_freq_mask, :]
    
    low_freq_summed_power = np.sum(low_freq_PSDs, axis=0)  # sum of PSD over low band per frame
    high_freq_summed_power = np.sum(high_freq_PSDs, axis=0)  # sum of PSD over high band per frame

    alpha_ratios = []
    speaking_times = []
    for j, t in enumerate(times):
        f0 = pitch.get_value_at_time(float(t))
        if np.isnan(f0):
            continue  # person is not speaking
        speaking_times.append(t)

        # Avoid divide-by-zero; skip frames where high-band energy is 0 or non-finite.
        denom = high_freq_summed_power[j]
        num = low_freq_summed_power[j]
        if (not np.isfinite(num)) or (not np.isfinite(denom)) or denom <= 0 or num <= 0:
            continue

        alpha_ratios.append(np.log10(num / denom))

    alpha_ratios = np.array(alpha_ratios)
    alpha_ratio_mean = float(np.mean(alpha_ratios)) if alpha_ratios.size else float("nan")
    speaking_times = np.array(speaking_times)
    
    if stats:
        # Timing:
        elapsed_sec = time.perf_counter() - start_time
        with open(stats_csv_file_name, "x", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "sound_path",
                "sample_rate_hz",
                "time_step",
                "window_length",
                "f_low1",
                "f_high1",
                "f_low2",
                "f_high2",
                "pitch_floor",
                "pitch_ceiling",
                "alpha_ratio_mean",
                "elapsed_seconds",
                "time_seconds",
                "alpha_ratio",
            ])
            if alpha_ratios.size == 0:
                writer.writerow([
                    os.path.basename(sound_path),
                    sampling_hz,
                    time_step,
                    window_length,
                    f_low1,
                    f_high1,
                    f_low2,
                    f_high2,
                    pitch_floor,
                    pitch_ceiling,
                    f"{alpha_ratio_mean:.6f}" if np.isfinite(alpha_ratio_mean) else "",
                    f"{elapsed_sec:.6f}",
                    "",
                    "",
                ])
            else:
                for i, (t, a) in enumerate(zip(speaking_times, alpha_ratios)):
                    writer.writerow([
                        os.path.basename(sound_path) if i == 0 else "",
                        sampling_hz if i == 0 else "",
                        time_step if i == 0 else "",
                        window_length if i == 0 else "",
                        f_low1 if i == 0 else "",
                        f_high1 if i == 0 else "",
                        f_low2 if i == 0 else "",
                        f_high2 if i == 0 else "",
                        pitch_floor if i == 0 else "",
                        pitch_ceiling if i == 0 else "",
                        (f"{alpha_ratio_mean:.6f}" if np.isfinite(alpha_ratio_mean) else "") if i == 0 else "",
                        f"{elapsed_sec:.6f}" if i == 0 else "",
                        f"{float(t):.6f}",
                        f"{float(a):.6f}",
                    ])
    return speaking_times, alpha_ratios, alpha_ratio_mean, sampling_hz
    

ts, ratios, alpha_ratio_mean, s_hz = alpha_ratio(snd, 'function_output_data')


#__________TESTING___________

print(alpha_ratio_mean)
# plt.scatter(ts, ratios, color='blue', label='Data Points')
# plt.axhline(y=alpha_ratio_mean, color='r', linestyle='-', label=f'Alpha ratio mean: {alpha_ratio_mean}')
# plt.title("Alpha ratios over time")
# plt.xlabel("Time (in seconds)")
# plt.ylabel("Alpha ratio")
# # plt.savefig("figures/alpha_ratios_high_pitched_audio.png", dpi=300, bbox_inches='tight')
# plt.show()
