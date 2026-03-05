import numpy as np
import parselmouth
from parselmouth.praat import call
import matplotlib.pyplot as plt
import time
import os
import csv


# BE CAREFUL BECAUSE LOUDNESS CAN CHANGE BASED ON MICROPHONE, AND LOUDNESS IS NOT ROBUST TO CONTEXT

# snd = "raw_audio/hoarse_test_voice.wav" # mean intensity was 50.64
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
    
    active_intensity_vals_mean = float(np.mean(active_intensity_vals)) if active_intensity_vals.size else float("nan")
    
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
                "time_seconds",
                "intensity_db",
            ])
            if active_times.size == 0:
                writer.writerow([
                    os.path.basename(sound_path),
                    sampling_hz,
                    time_step,
                    pitch_floor,
                    subtract_mean,
                    activity_threshold_db,
                    f"{active_intensity_vals_mean:.6f}" if np.isfinite(active_intensity_vals_mean) else "",
                    f"{elapsed_sec:.6f}",
                    "",
                    "",
                ])
            else:
                for i, (t, v) in enumerate(zip(active_times, active_intensity_vals)):
                    writer.writerow([
                        os.path.basename(sound_path) if i == 0 else "",
                        sampling_hz if i == 0 else "",
                        time_step if i == 0 else "",
                        pitch_floor if i == 0 else "",
                        subtract_mean if i == 0 else "",
                        activity_threshold_db if i == 0 else "",
                        (f"{active_intensity_vals_mean:.6f}" if np.isfinite(active_intensity_vals_mean) else "") if i == 0 else "",
                        f"{elapsed_sec:.6f}" if i == 0 else "",
                        f"{t:.6f}",
                        f"{v:.3f}",
                    ])
    
    return active_times, active_intensity_vals, active_intensity_vals_mean, sampling_hz


def save_loudness_plot(
    ts: np.ndarray,
    intensity_db: np.ndarray,
    mean_intensity_db: float,
    output_path: str,
):
    """Save loudness plot in the same format as the TESTING section."""
    if ts.size == 0 or intensity_db.size == 0:
        return

    plt.figure()
    plt.scatter(ts, intensity_db, color='blue', label='Data Points')
    plt.axhline(y=mean_intensity_db, color='r', linestyle='-',
                label=f'Intensity mean: {mean_intensity_db}')
    plt.title("Intensities over time")
    plt.xlabel("Time (in seconds)")
    plt.ylabel("Intensity (dB)")
    plt.legend()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    patient_raw_data_directory = '/data_store2/resection/neuropsych_video/presidio/Stage2/PR05/home/'

    loudness_csv_output_directory = '/userdata/msharma/sub-PR05_stage-2_audio-audiotype_preproc_loudness_metadata'
    loudness_plot_output_directory = '/userdata/msharma/sub-PR05_stage-2_audio-audiotype_preproc_loudness_plots'

    os.makedirs(loudness_csv_output_directory, exist_ok=True)
    os.makedirs(loudness_plot_output_directory, exist_ok=True)

    for num in range(579, 611):
        audio_name_without_wav = str(num)
        audio_name = audio_name_without_wav + '_audio.wav'
        sound_path = os.path.join(patient_raw_data_directory, audio_name)

        if not os.path.exists(sound_path):
            continue

        ts, intensity_db, mean_intensity_db, sampling_hz = loudness_in_db(
            sound_path,
            loudness_csv_output_directory,
        )

        plot_path = os.path.join(
            loudness_plot_output_directory,
            'sub-PR05_stage-2_audio-audiotype_preproc_' + audio_name_without_wav + '_loudness.png'
        )
        save_loudness_plot(ts, intensity_db, mean_intensity_db, plot_path)


if __name__ == "__main__":
    main()


    
# ts, intensity, mean_intensity, s_hz = loudness_in_db(snd, 'function_output_data')


#__________TESTING___________

# print(mean_intensity)
# plt.scatter(ts, intensity, color='blue', label='Data Points')
# plt.axhline(y=mean_intensity, color='r', linestyle='-', label=f'Intensity mean: {mean_intensity}')
# plt.title("Intensities over time")
# plt.xlabel("Time (in seconds)")
# plt.ylabel("Intensity (dB)")
# plt.savefig("figures/loudnesshoarsevoice.png", dpi=300, bbox_inches='tight')
# plt.show()