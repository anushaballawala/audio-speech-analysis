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

    pitch_mean = float(np.mean(nonzero_f0_values)) if nonzero_f0_values.size else float("nan")

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
                "time_seconds",
                "f0_hz",
            ])
            if nonzero_xs.size == 0:
                # No nonzero pitch frames: write a single row with metadata + blanks for time-varying columns.
                writer.writerow([
                    os.path.basename(sound_path),
                    sampling_hz,
                    time_step,
                    pitch_floor,
                    pitch_ceiling,
                    f0_lstsq_slope,
                    f"{pitch_mean:.6f}" if np.isfinite(pitch_mean) else "",
                    f"{elapsed_sec:.6f}",
                    "",
                    "",
                ])
            else:
                for i, (t, f0) in enumerate(zip(nonzero_xs, nonzero_f0_values)):
                    writer.writerow([
                        os.path.basename(sound_path) if i == 0 else "",
                        sampling_hz if i == 0 else "",
                        time_step if i == 0 else "",
                        pitch_floor if i == 0 else "",
                        pitch_ceiling if i == 0 else "",
                        f"{f0_lstsq_slope:.6f}" if i == 0 else "",
                        (f"{pitch_mean:.6f}" if np.isfinite(pitch_mean) else "") if i == 0 else "",
                        f"{elapsed_sec:.6f}" if i == 0 else "",
                        f"{t:.6f}",
                        f"{f0:.3f}",
                    ])
    
    return nonzero_xs, nonzero_f0_values, f0_lstsq_slope, f0_lstsq_intercept, sampling_hz


# Save a pitch scatter plot + line of best fit in the same format as the TESTING section.
def save_pitch_plot(
    nonzero_xs: np.ndarray,
    nonzero_f0_values: np.ndarray,
    f0_lstsq_slope: float,
    f0_lstsq_intercept: float,
    output_path: str,
):
    """Save a pitch scatter plot + line of best fit in the same format as the TESTING section."""
    if nonzero_xs.size == 0 or nonzero_f0_values.size == 0:
        return

    line_y_values = f0_lstsq_slope * nonzero_xs + f0_lstsq_intercept
    plt.figure()
    plt.scatter(nonzero_xs, nonzero_f0_values, color='blue', label='Data Points')
    plt.plot(nonzero_xs, line_y_values, color='red', linestyle='-', label='Line of Best Fit')
    plt.title("Nonzero f0 values")
    plt.xlabel("Time (in seconds)")
    plt.ylabel("Hertz")
    plt.legend()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


 # nonzero_xs, nonzero_f0_values, f0_lstsq_slope, f0_lstsq_intercept, s_hz = pitches(snd, 'function_output_data')

def main():
    # Input directory and naming convention must match the provided patient script
    patient_raw_data_directory = '/data_store2/resection/neuropsych_video/presidio/Stage2/PR05/home/'

    # Output directories (as requested)
    pitch_csv_output_directory = '/userdata/msharma/sub-PR05_stage-2_audio-audiotype_preproc_pitch_metadata'
    pitch_plot_output_directory = '/userdata/msharma/sub-PR05_stage-2_audio-audiotype_preproc_pitch_plots'

    os.makedirs(pitch_csv_output_directory, exist_ok=True)
    os.makedirs(pitch_plot_output_directory, exist_ok=True)

    for num in range(579, 611):
        audio_name_without_wav = str(num)
        audio_name = audio_name_without_wav + '_audio.wav'
        sound_path = os.path.join(patient_raw_data_directory, audio_name)

        if not os.path.exists(sound_path):
            continue

        nonzero_xs, nonzero_f0_values, f0_lstsq_slope, f0_lstsq_intercept, s_hz = pitches(
            sound_path,
            pitch_csv_output_directory,
        )

        plot_path = os.path.join(
            pitch_plot_output_directory,
            'sub-PR05_stage-2_audio-audiotype_preproc_' + audio_name_without_wav + '_pitch.png'
        )
        save_pitch_plot(nonzero_xs, nonzero_f0_values, f0_lstsq_slope, f0_lstsq_intercept, plot_path)


if __name__ == "__main__":
    main()


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
