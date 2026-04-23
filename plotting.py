import parselmouth
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

sns.set_theme()
plt.rcParams['figure.dpi'] = 100


def save_waveform_plot(audio_path, output_folder="/userdata/msharma/PR05_audio_plots", name="PR05"):
    """
    Saves a waveform plot for audio file.

    Parameters:
        audio_path (str): Path to the audio file.
        output_folder (str): Folder where plots will be saved.
    """
    # Load sound
    snd = parselmouth.Sound(audio_path)

    # Create output folder if needed
    os.makedirs(output_folder, exist_ok=True)

    # Create plot
    plt.figure()
    plt.plot(snd.xs(), snd.values.T)
    plt.xlim([snd.xmin, snd.xmax])
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(f"Waveform: {os.path.basename(audio_path)}")

    # Generate output filename
    filename = os.path.splitext(os.path.basename(audio_path))[0]
    output_path = os.path.join(output_folder, f"{filename}_{name}_waveform.png")

    # Save and close
    plt.savefig(output_path)
    plt.close()

    # print(f"Saved: {output_path}")
    
def save_spectrogram_plot(audio_path,
                          output_folder="/userdata/msharma/PR05_audio_plots",
                          name="PR05"):
    """
    Saves a spectrogram plot for an audio file.

    Parameters:
        audio_path (str): Path to the audio file.
        output_folder (str): Folder where plots will be saved.
        name (str): Custom name tag for output file.
    """
    # Load sound
    snd = parselmouth.Sound(audio_path)

    # Create spectrogram
    spectrogram = snd.to_spectrogram()

    # Extract values
    X = spectrogram.x_grid()        # time axis
    Y = spectrogram.y_grid()        # frequency axis
    Z = spectrogram.values          # intensity values

    # Convert to dB
    Z_db = 10 * np.log10(np.maximum(Z, 1e-10))

    # Create output folder if needed
    os.makedirs(output_folder, exist_ok=True)

    # Plot
    plt.figure()
    plt.pcolormesh(X, Y, Z_db, shading='auto')
    plt.ylim(0, 5000)  # limit to speech range (adjust if needed)
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title(f"Spectrogram: {os.path.basename(audio_path)}")
    plt.colorbar(label="Intensity [dB]")

    # Generate output filename
    filename = os.path.splitext(os.path.basename(audio_path))[0]
    output_path = os.path.join(output_folder,
                               f"{filename}_{name}_spectrogram.png")

    # Save and close
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_spectrum_plot(audio_path,
                       output_folder="/userdata/msharma/PR05_audio_plots",
                       name="PR05",
                       freq_max=5000.0,
                       snippet_duration=0.1,
                       pitch_floor=115.0):
    """
    Saves a power spectrum plot from a short snippet of active speech.

    Finds the peak-intensity moment in the recording, extracts a short
    window (~snippet_duration seconds) centred there, and computes the
    FFT of that snippet.  This avoids smearing speech harmonics with
    silence/noise the way a full-signal FFT would.

    Parameters:
        audio_path (str): Path to the audio file.
        output_folder (str): Folder where plots will be saved.
        name (str): Custom name tag for output file.
        freq_max (float): Upper frequency limit for the plot (Hz).
        snippet_duration (float): Length of the analysis window in seconds
                                  (default 100 ms — long enough for ~1
                                  glottal cycle at the lowest expected f0).
        pitch_floor (float): Minimum pitch used by the intensity algorithm
                             to locate active speech.
    """
    from parselmouth.praat import call as praat_call

    snd = parselmouth.Sound(audio_path)

    # ── Find the loudest moment using Praat intensity ──
    intensity = praat_call(snd, "To Intensity...", pitch_floor, 0.01, True)
    vals = intensity.values[0, :]
    times = intensity.xs()

    if vals.size == 0:
        return

    peak_idx = int(np.argmax(vals))
    peak_time = times[peak_idx]

    # ── Extract a centred snippet, clamped to signal bounds ──
    half = snippet_duration / 2.0
    t_start = max(snd.xmin, peak_time - half)
    t_end = min(snd.xmax, peak_time + half)
    snippet = snd.extract_part(t_start, t_end,
                               parselmouth.WindowShape.HAMMING, 1.0, False)

    # ── FFT of the snippet ──
    spectrum = snippet.to_spectrum()
    freqs = spectrum.xs()
    real = spectrum.values[0, :]
    imag = spectrum.values[1, :]
    power = real**2 + imag**2
    power_db = 10 * np.log10(np.maximum(power, 1e-20))

    # Restrict to [0, freq_max]
    mask = freqs <= freq_max
    freqs = freqs[mask]
    power_db = power_db[mask]

    os.makedirs(output_folder, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, power_db, color='steelblue', linewidth=0.6)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power [dB]")
    plt.title(f"Spectrum ({snippet_duration*1e3:.0f} ms @ {peak_time:.2f} s): "
              f"{os.path.basename(audio_path)}")
    plt.xlim(0, freq_max)
    plt.grid(axis='both', alpha=0.3)

    filename = os.path.splitext(os.path.basename(audio_path))[0]
    output_path = os.path.join(output_folder,
                               f"{filename}_{name}_spectrum.png")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():

    # patient_raw_data_directory = '/data_store2/resection/neuropsych_video/presidio/Stage2/PR05/home/'
    patient_raw_data_directory = '/data_store2/resection/neuropsych_video/presidio/Stage2/PR05/home/sub-PR05_stage-2_audio-athome_signal-preproc/'
    plot_output_directory = '/userdata/msharma/sub-PR05_stage-2_audio-audiotype_raw_audio_plots'
    
    for num in range(579, 1067):
        # audio_name = str(num) + '_audio.wav'
        audio_name = 'sub-PR05_stage-2_audio-athome_signal-preproc_' + str(num) + '.wav'
        audio_path = os.path.join(patient_raw_data_directory, audio_name)
        if os.path.exists(audio_path):
            save_waveform_plot(audio_path, plot_output_directory)
            save_spectrogram_plot(audio_path, plot_output_directory)
            save_spectrum_plot(audio_path, plot_output_directory)

if __name__ == "__main__":
    main()