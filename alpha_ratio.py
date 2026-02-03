import parselmouth
import numpy as np
from numpy.linalg import lstsq
import matplotlib.pyplot as plt

# # snd = "raw_audio/hoarse_test_voice.wav"
snd = "raw_audio/testsoundmono.mp3" # gives mean alpha ratio 5976.599 (w/out log10), 2.079 w/ log10
# snd = "raw_audio/high_pitch.wav" # gives mean alpha ratio of 112.417 (w/out log10), 1.516 w/ log10


def alpha_ratio(
    sound_path: str,
    time_step: float = 0.01, 
    window_length: float = 0.025,
    f_low1: float = 50.0,
    f_high1: float = 1000.0,
    f_low2: float = 1000.0,
    f_high2: float = 5000.0
):
    """
     Linear alpha ratio per frame:
         alpha(t) = log10(sum(E[50-1000 Hz]) / sum(E[1000-5000 Hz]))
     Returns: times (s), alpha_ratio_per_frame, alpha_ratio_mean
    """
    
    sound = parselmouth.Sound(sound_path)
    sampling_hz = sound.sampling_frequency
    
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
    
    low_freq_summed_power = np.sum(low_freq_PSDs, axis = 0) # really sum of power spectrum densities (in units Pa^2/Hz)
    high_freq_summed_power = np.sum(high_freq_PSDs, axis = 0)
    
    alpha_ratios = np.log10(low_freq_summed_power / high_freq_summed_power) # ok to use direct summed powers since it gives the same ratio as energies since frequency bins cancel out
    
    alpha_ratio_mean = np.mean(alpha_ratios)
    
    return times, alpha_ratios, alpha_ratio_mean, sampling_hz
    

ts, ratios, alpha_ratio_mean, s_hz = alpha_ratio(snd)
print(alpha_ratio_mean)
# plt.scatter(ts, ratios, color='blue', label='Data Points')
# plt.axhline(y=alpha_ratio_mean, color='r', linestyle='-', label=f'Alpha ratio mean: {alpha_ratio_mean}')
# plt.title("Alpha ratios over time")
# plt.xlabel("Time (in seconds)")
# plt.ylabel("Alpha ratio")
# plt.savefig("figures/alpha_ratios_high_pitched_audio.png", dpi=300, bbox_inches='tight')
# plt.show()
