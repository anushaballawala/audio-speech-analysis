import numpy as np
import parselmouth
from parselmouth.praat import call
import matplotlib.pyplot as plt
import time
import os
import csv


#RELATIVE ENERGY FOR FORMANT 3 SHOULD BE MORE NEGATIVE WITH DEPRESSED PATIENTS

# snd = "raw_audio/hoarse_test_voice.wav" #  THIS IS AN OUTLIER SINCE THE HAORSE TEST VOICE FUNDAMENTAL FREQUENCY WAS ALL OVER THE PLACE. WE WILL NEED ANOTHER METHOD TO EXTRACT WHEN A PERSON IS TALKING.
# snd = "raw_audio/testsoundmono.mp3" # mean_rel_energy_f_3: -43.187290370551345
# snd = "raw_audio/high_pitch.wav" # mean_rel_energy_f_3: -37.68999286239633 (CORRECT: EXPECTED VALUE TO BE SMALLER IN MAGNITUDE --> ratio is larger --> more f3, which is correct!)

def relative_energy_formant(
    sound_path: str, 
    csv_folder_name: str,
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
    return_db: bool = True,  # if True, returns 10*log_10(relative_energy)
    stats: bool = True
):
    """
    Returns:
      times: np.ndarray (s)
      rel_energy_f_i: np.ndarray (linear ratio, or dB if return_db=True)
      mean_rel_energy_f_i: float
    """
    if(stats):
        start_time = time.perf_counter()
        wav_base = os.path.splitext(os.path.basename(sound_path))[0]
        func_name = relative_energy_formant.__name__
        stats_csv_file_name = f"{csv_folder_name}/{wav_base}_{func_name}.csv"
    
    sound = parselmouth.Sound(sound_path)
    
    sampling_hz = sound.sampling_frequency
    
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
    
    speaking_times = []
    for j, t in enumerate(times):
      f0 = pitch.get_value_at_time(float(t))
      if np.isnan(f0):
        continue #person is not speaking
      
      f_i = call(formant_freqs, "Get value at time...", formant, float(t), "Hertz", "Linear") # value of formant_i in hz at time t
      if not np.isfinite(f_i) or f_i < 0 or f_i > max_freq_hz:
        continue
      
      f_i_mask = (freqs >= (f_i - half_bandwidth_hz)) & (freqs <= (f_i + half_bandwidth_hz))
      
      f_i_frequency_PSDs = frequency_PSDs[f_i_mask, j]
      
      f_i_summed_power = np.sum(f_i_frequency_PSDs)
      
      frame_summed_power = np.sum(frequency_PSDs[:, j])
  
      relative_energies.append(f_i_summed_power / frame_summed_power) # ok to use direct summed powers since it gives the same ratio as energies since frequency bins cancel out
      speaking_times.append(t)
    
    if(return_db):
      relative_energies = 10 * np.log10(relative_energies)
    mean_rel_energy_f_i = float(np.mean(relative_energies)) if len(relative_energies) else float("nan")
    
    if stats:
        # Timing:
        elapsed_sec = time.perf_counter() - start_time
        with open(stats_csv_file_name, "x", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "sound_path",
                "sample_rate_hz",
                "time_step",
                "max_formant_hz",
                "n_formants",
                "formant_window_length",
                "pre_emphasis_from_hz",
                "spec_window_length",
                "max_freq_hz",
                "pitch_floor",
                "pitch_ceiling",
                "f_i_bandwidth_hz",
                "return_db",
                "mean_rel_energy_f_i",
                "elapsed_seconds",
                "time_seconds",
                "relative_energy",
            ])
            if len(relative_energies) == 0:
                writer.writerow([
                    os.path.basename(sound_path),
                    sampling_hz,
                    time_step,
                    max_formant_hz,
                    n_formants,
                    formant_window_length,
                    pre_emphasis_from_hz,
                    spec_window_length,
                    max_freq_hz,
                    pitch_floor,
                    pitch_ceiling,
                    f_i_bandwidth_hz,
                    return_db,
                    f"{mean_rel_energy_f_i:.6f}" if np.isfinite(mean_rel_energy_f_i) else "",
                    f"{elapsed_sec:.6f}",
                    "",
                    "",
                ])
            else:
                for i, (t, re) in enumerate(zip(speaking_times, relative_energies)):
                    writer.writerow([
                        os.path.basename(sound_path) if i == 0 else "",
                        sampling_hz if i == 0 else "",
                        time_step if i == 0 else "",
                        max_formant_hz if i == 0 else "",
                        n_formants if i == 0 else "",
                        formant_window_length if i == 0 else "",
                        pre_emphasis_from_hz if i == 0 else "",
                        spec_window_length if i == 0 else "",
                        max_freq_hz if i == 0 else "",
                        pitch_floor if i == 0 else "",
                        pitch_ceiling if i == 0 else "",
                        f_i_bandwidth_hz if i == 0 else "",
                        return_db if i == 0 else "",
                        (f"{mean_rel_energy_f_i:.6f}" if np.isfinite(mean_rel_energy_f_i) else "") if i == 0 else "",
                        f"{elapsed_sec:.6f}" if i == 0 else "",
                        f"{float(t):.6f}",
                        f"{float(re):.6f}",
                    ])
    
    return np.array(speaking_times), np.array(relative_energies), mean_rel_energy_f_i, sampling_hz
      
    


def save_f3_plot(
    ts: np.ndarray,
    relative_energies: np.ndarray,
    mean_rel_energy_f_3: float,
    output_path: str,
):
    """Save the f3 relative energy plot in the same format as the TESTING section."""
    if ts.size == 0 or relative_energies.size == 0:
        return

    plt.figure()
    plt.scatter(ts, relative_energies, color='blue', label='Data Points')
    plt.axhline(y=mean_rel_energy_f_3, color='r', linestyle='-',
                label=f'f3 relative energy mean: {mean_rel_energy_f_3}')
    plt.title("f3 relative energy over time")
    plt.xlabel("Time (in seconds)")
    plt.ylabel("f3 relative energy")
    plt.legend()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    patient_raw_data_directory = '/data_store2/resection/neuropsych_video/presidio/Stage2/PR05/home/'

    f3_csv_output_directory = '/userdata/msharma/sub-PR05_stage-2_audio-audiotype_preproc_f3_metadata'
    f3_plot_output_directory = '/userdata/msharma/sub-PR05_stage-2_audio-audiotype_preproc_f3_plots'

    os.makedirs(f3_csv_output_directory, exist_ok=True)
    os.makedirs(f3_plot_output_directory, exist_ok=True)

    for num in range(579, 611):
        audio_name_without_wav = str(num)
        audio_name = audio_name_without_wav + '_audio.wav'
        sound_path = os.path.join(patient_raw_data_directory, audio_name)

        if not os.path.exists(sound_path):
            continue

        ts, relative_energies, mean_rel_energy_f_3, sampling_hz = relative_energy_formant(
            sound_path,
            f3_csv_output_directory,
            3,
        )

        plot_path = os.path.join(
            f3_plot_output_directory,
            'sub-PR05_stage-2_audio-audiotype_preproc_' + audio_name_without_wav + '_f3.png'
        )
        save_f3_plot(ts, relative_energies, mean_rel_energy_f_3, plot_path)


if __name__ == "__main__":
    main()

#__________TESTING___________
# print(mean_rel_energy_f_3)

# plt.scatter(ts, relative_energies, color='blue', label='Data Points')
# plt.axhline(y=mean_rel_energy_f_3, color='r', linestyle='-', label=f'f3 relative energy mean: {mean_rel_energy_f_3}')
# plt.title("f3 relative energy over time")
# plt.xlabel("Time (in seconds)")
# plt.ylabel("f3 relative energy")
# # plt.savefig("figures/f3_rel_energies_normal_audio.png", dpi=300, bbox_inches='tight')
# plt.show()