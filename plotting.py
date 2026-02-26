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
    import parselmouth
    import matplotlib.pyplot as plt
    import numpy as np
    import os

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

def main():
    audio_folder = "raw_audio"
    output_folder = "plots"
    save_waveform_plot('raw_audio/high_pitch.wav', 'plots', '')

    patient_raw_data_directory = '/data_store2/resection/neuropsych_video/presidio/Stage2/PR05/home/'
    plot_output_directory = '/userdata/msharma/sub-PR05_stage-2_audio-audiotype_raw_audio_plots'
    # patient_function_output_directory = '/userdata/msharma/sub-PR05_stage-2_audio-audiotype_preproc_metadata'
    
    for num in range(579, 611):
        audio_name = str(num) + '_audio.wav'
        audio_name_without_wav = str(num)
        if os.path.exists(patient_raw_data_directory + audio_name):
            save_waveform_plot(patient_raw_data_directory + audio_name, plot_output_directory)
            save_spectrogram_plot(patient_raw_data_directory + audio_name, plot_output_directory)

if __name__ == "__main__":
    main()