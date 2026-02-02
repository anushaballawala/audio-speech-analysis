import numpy as np
import parselmouth
from parselmouth.praat import call
import matplotlib.pyplot as plt


# BE CAREFUL BECAUSE LOUDNESS CAN CHANGE BASED ON MICROPHONE, AND LOUDNESS IS NOT ROBUST TO CONTEXT

snd = parselmouth.Sound("raw_audio/hoarse_test_voice.wav") # mean intensity was 50.64
# snd = parselmouth.Sound("raw_audio/testsoundmono.mp3") # mean intensity was 57.78
# snd = parselmouth.Sound("raw_audio/high_pitch.wav") 

def loudness_in_db(
    sound: parselmouth.Sound,
    time_step: float = 0.01,
    pitch_floor: float = 75.0,
    subtract_mean: bool = True,
    activity_threshold_db: float = 40.0 # CAN ADJUST THIS VALUE (NORMALLY AROUND 40 BUT I SET IT TO 20) #TODO ASK IF I SHOULD MASK FOR ACTIVE NOISE WINDOWS USING NONZERO PITCH VALUES (CURRENTLY USING NOISE THRESHOLD WHICH IS NORMAL)
):
    """
    Mean intensity (dB) over active frames only.
    Returns times (s), intensity_per_frame, mean_intensity
    """
    intensity = call(sound, "To Intensity...", pitch_floor, time_step, subtract_mean) #
    vals = intensity.values[0, :]
    times = intensity.xs()
    
    active_mask = vals >= activity_threshold_db
    active_times = times[active_mask]
    active_intensity_vals = vals[active_mask]
    
    active_intensity_vals_mean = np.mean(active_intensity_vals)
    
    return active_times, active_intensity_vals, active_intensity_vals_mean
    
    
    
ts, intensity, mean_intensity = loudness_in_db(snd)
print(mean_intensity)

# plt.scatter(ts, intensity, color='blue', label='Data Points')
# plt.axhline(y=mean_intensity, color='r', linestyle='-', label=f'Intensity mean: {mean_intensity}')
# plt.title("Intensities over time")
# plt.xlabel("Time (in seconds)")
# plt.ylabel("Intensity (dB)")
# plt.savefig("figures/loudnesshoarsevoice.png", dpi=300, bbox_inches='tight')
# plt.show()

    