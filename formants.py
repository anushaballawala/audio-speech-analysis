import numpy as np
import parselmouth
from parselmouth.praat import call
import matplotlib.pyplot as plt


#RELATIVE ENERGY FOR FORMANT 3 SHOULD BE MORE NEGATIVE WITH DEPRESSED PATIENTS

# snd = parselmouth.Sound("raw_audio/hoarse_test_voice.wav") #  THIS IS AN OUTLIER SINCE THE HAORSE TEST VOICE FUNDAMENTAL FREQUENCY WAS ALL OVER THE PLACE. WE WILL NEED ANOTHER METHOD TO EXTRACT WHEN A PERSON IS TALKING.
snd = parselmouth.Sound("raw_audio/testsoundmono.mp3") # mean_rel_energy_f_3: -37.54012983676498
# snd = parselmouth.Sound("raw_audio/high_pitch.wav") # mean_rel_energy_f_3: -35.545650404064226 (CORRECT: EXPECTED VALUE TO BE SMALLER IN MAGNITUDE --> ratio is larger --> more f3, which is correct!)

def relative_energy_formant(
    sound: parselmouth.Sound,
    formant: int, # the formant frequency whose relative energy is to be extracted
    time_step: float = 0.01,
    max_formant_hz: float = 5500.0,
    n_formants: int = 5,
    formant_window_length: float = 0.025,
    pre_emphasis_from_hz: float = 50.0,
    # Spectrogram settings (used for energy)
    spec_window_length: float = 0.025,
    max_freq_hz: float = 5000.0, # MAX HZ ALLOWED CAP ALLOWED to be counted as part of f_i (most of time f_i extracted will never reach this)
    pitch_floor: float = 75.0, # PITCH FLOOR AND CEILING ARE FOR DETERMINING FRAMES WHEN A PERSON IS SPEAKING CALCULATED USING NONZERO F0 VALUES. FIXME ASK ABOUT PITCH FLOOR/CELING VALUES TO BE USED; PITCH MIN WAS AN IMPORTANT FEATURE IN DETERMINING DIFFERENCES (w/ 27.5 min pitch was 28.3 With 75 min pitch was around 96.5); 27.5 looks like it creates outliers
    pitch_ceiling: float = 500.0,
    f_i_bandwidth_hz: float = -1,   # get energies +/- f_i_bandwidth_hz/2 Hz around f_i to capture all f_i energy; for defaults, use -1. 
    return_db: bool = True  # if True, returns 10*log10(relative_energy)
):
    """
    Returns:
      times: np.ndarray (s)
      rel_energy_f_i: np.ndarray (linear ratio, or dB if return_db=True)
      mean_rel_energy_f_i: float
    """
    
    # default values based on paper for bandwidth hz: (Link to paper: https://www.isca-archive.org/eurospeech_1999/karlsson99_eurospeech.pdf)
    formant_to_bandwidth_hz = {1: 60, 2: 90, 3: 150, 4: 200}
    if (f_i_bandwidth_hz == -1):
      f_i_bandwidth_hz = formant_to_bandwidth_hz.get(formant, 100) #gets the bandwidth value for the ith formant, and defaults to 100 if it's not in the dictionary
    
    formant_freqs = call(
        sound,
        "To Formant (burg)...",
        time_step,
        n_formants,
        max_formant_hz,
        formant_window_length,
        pre_emphasis_from_hz,
    )
    
    spectrogram = sound.to_spectrogram(
         window_length=spec_window_length,
         time_step=time_step,
         maximum_frequency=max_freq_hz
    )
    
    pitch = sound.to_pitch(time_step=time_step, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling) 
    
    frequency_PSDs = spectrogram.values # Power spectrum density values (PSD)
    freqs = spectrogram.ys()
    times = spectrogram.xs()
    
    half_bandwidth_hz = f_i_bandwidth_hz / 2
    
    relative_energies = []
    
    for j, t in enumerate(times):
      f0 = pitch.get_value_at_time(float(t))
      if f0 <= 0:
        continue #person is not speaking
      
      f_i = call(formant_freqs, "Get value at time...", formant, float(t), "Hertz", "Linear") # value of formant_i in hz at time t
      if not np.isfinite(f_i) or f_i < 0 or f_i > max_freq_hz:
        continue
      
      f_i_mask = (freqs >= (f_i - half_bandwidth_hz)) & (freqs <= (f_i + half_bandwidth_hz))
      
      f_i_frequency_PSDs = frequency_PSDs[f_i_mask, j]
      
      f_i_summed_power = np.sum(f_i_frequency_PSDs)
      
      frame_summed_power = np.sum(frequency_PSDs[:, j])
      
      relative_energies.append(f_i_summed_power / frame_summed_power) # ok to use direct summed powers since it gives the same ratio as energies since frequency bins cancel out
    
    if(return_db):
      relative_energies = 10 * np.log10(relative_energies)
    mean_rel_energy_f_i = np.mean(relative_energies)
    
    return times, relative_energies, mean_rel_energy_f_i
      
    
ts, relative_energies, mean_rel_energy_f_3 = relative_energy_formant(snd, 3)


#__________TESTING___________
# print(mean_rel_energy_f_3)

# plt.scatter(ts, relative_energies, color='blue', label='Data Points')
# plt.axhline(y=mean_rel_energy_f_3, color='r', linestyle='-', label=f'f3 relative energy mean: {mean_rel_energy_f_3}')
# plt.title("f3 relative energy over time")
# plt.xlabel("Time (in seconds)")
# plt.ylabel("f3 relative energy")
# plt.savefig("figures/f3_rel_energies_normal_audio.png", dpi=300, bbox_inches='tight')
# plt.show()