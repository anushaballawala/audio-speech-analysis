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
    pitch_floor: float = 115.0,
    subtract_mean: bool = True,
    activity_threshold_db: float = 40.0, # CAN ADJUST THIS VALUE (NORMALLY AROUND 40)
    stats: bool = True
):
    """
    Mean intensity (dB) over active frames only.
    Returns:
      active_times: np.ndarray (s)
      active_intensity_vals: np.ndarray
      active_intensity_vals_mean: float
      sampling_hz: float
      intensity_std: float          — std dev of active intensity values
      intensity_median: float       — median of active intensity values
      intensity_iqr: float          — interquartile range (Q3 - Q1)
      intensity_q1: float           — 25th percentile
      intensity_q3: float           — 75th percentile
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

    # ── New spread / dispersion metrics ──
    if active_intensity_vals.size > 0:
        intensity_std = float(np.std(active_intensity_vals, ddof=1)) if active_intensity_vals.size > 1 else 0.0
        intensity_median = float(np.median(active_intensity_vals))
        intensity_q1 = float(np.percentile(active_intensity_vals, 25))
        intensity_q3 = float(np.percentile(active_intensity_vals, 75))
        intensity_iqr = intensity_q3 - intensity_q1
    else:
        intensity_std = float("nan")
        intensity_median = float("nan")
        intensity_q1 = float("nan")
        intensity_q3 = float("nan")
        intensity_iqr = float("nan")
    
    if stats:
        # Timing:
        elapsed_sec = time.perf_counter() - start_time
        fmt = lambda v: f"{v:.6f}" if np.isfinite(v) else ""
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
                "intensity_std",
                "intensity_median",
                "intensity_q1",
                "intensity_q3",
                "intensity_iqr",
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
                    fmt(active_intensity_vals_mean),
                    fmt(intensity_std),
                    fmt(intensity_median),
                    fmt(intensity_q1),
                    fmt(intensity_q3),
                    fmt(intensity_iqr),
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
                        fmt(active_intensity_vals_mean) if i == 0 else "",
                        fmt(intensity_std) if i == 0 else "",
                        fmt(intensity_median) if i == 0 else "",
                        fmt(intensity_q1) if i == 0 else "",
                        fmt(intensity_q3) if i == 0 else "",
                        fmt(intensity_iqr) if i == 0 else "",
                        f"{elapsed_sec:.6f}" if i == 0 else "",
                        f"{t:.6f}",
                        f"{v:.3f}",
                    ])
    
    return (active_times, active_intensity_vals, active_intensity_vals_mean, sampling_hz,
            intensity_std, intensity_median, intensity_iqr, intensity_q1, intensity_q3)


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


def save_summary_point_plot(
    recording_labels: list,
    values: list,
    ylabel: str,
    title: str,
    output_path: str,
):
    """
    Save a summary point plot of a single metric across all recordings.

    Each point represents one recording's value, plotted sequentially
    along the x-axis — one dot per audio file, no error bars.

    Args:
        recording_labels: list of recording identifiers (e.g. file numbers).
        values: list of metric values (one per recording).
        ylabel: y-axis label.
        title: plot title.
        output_path: where to save the figure.
    """
    if len(values) == 0:
        return

    labels = np.array(recording_labels)
    vals = np.array(values, dtype=float)

    # Drop any recordings that returned NaN
    valid = np.isfinite(vals)
    labels = labels[valid]
    vals = vals[valid]

    if vals.size == 0:
        return

    x_positions = np.arange(len(vals))
    grand_mean = float(np.mean(vals))

    fig, ax = plt.subplots(figsize=(max(10, len(vals) * 0.06), 5))

    ax.scatter(x_positions, vals, color='steelblue', s=20, zorder=3)
    ax.axhline(y=grand_mean, color='crimson', linestyle='--', linewidth=1,
               label=f'Grand mean: {grand_mean:.4f}')

    # X-tick labels: show a subset to avoid overlap when many recordings
    n = len(labels)
    if n > 40:
        tick_step = max(1, n // 20)  # show ~20 evenly-spaced ticks
        tick_indices = list(range(0, n, tick_step))
        ax.set_xticks([x_positions[i] for i in tick_indices])
        ax.set_xticklabels([labels[i] for i in tick_indices],
                           rotation=45, ha='right', fontsize=8)
    else:
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)

    ax.set_xlabel("Recording")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Summary plot saved to {output_path}")


def save_summary_errorbar_plot(
    recording_labels: list,
    centers: list,
    lower_bars: list,
    upper_bars: list,
    ylabel: str,
    title: str,
    output_path: str,
    bar_label: str = "±1 SD",
):
    """
    Save a summary plot with error bars (glacier-style).

    Each recording gets a point at `centers[i]` with asymmetric error bars
    extending down by `lower_bars[i]` and up by `upper_bars[i]`.

    Args:
        recording_labels: list of recording identifiers.
        centers: central value per recording (e.g. mean intensity).
        lower_bars: distance from center to lower bar (positive number).
        upper_bars: distance from center to upper bar (positive number).
        ylabel: y-axis label.
        title: plot title.
        output_path: where to save the figure.
        bar_label: label for the error bars in the legend.
    """
    if len(centers) == 0:
        return

    labels = np.array(recording_labels)
    ctrs = np.array(centers, dtype=float)
    lo = np.array(lower_bars, dtype=float)
    hi = np.array(upper_bars, dtype=float)

    valid = np.isfinite(ctrs)
    labels, ctrs, lo, hi = labels[valid], ctrs[valid], lo[valid], hi[valid]
    if ctrs.size == 0:
        return

    x_positions = np.arange(len(ctrs))
    grand_mean = float(np.mean(ctrs))

    fig, ax = plt.subplots(figsize=(max(12, len(ctrs) * 0.06), 6))

    ax.errorbar(
        x_positions, ctrs,
        yerr=np.array([lo, hi]),       # [lower, upper] distances
        fmt='o', markersize=4,
        color='steelblue', ecolor='steelblue', elinewidth=1.0,
        capsize=2, capthick=0.8, alpha=0.85,
        label=bar_label,
    )

    ax.axhline(y=grand_mean, color='crimson', linestyle='--', linewidth=1,
               label=f'Grand mean: {grand_mean:.4f}')

    n = len(labels)
    if n > 40:
        tick_step = max(1, n // 20)
        tick_indices = list(range(0, n, tick_step))
        ax.set_xticks([x_positions[i] for i in tick_indices])
        ax.set_xticklabels([labels[i] for i in tick_indices],
                           rotation=45, ha='right', fontsize=8)
    else:
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)

    ax.set_xlabel("Recording")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Error-bar summary plot saved to {output_path}")


def main():
    patient_preproc_data_directory = '/data_store2/resection/neuropsych_video/presidio/Stage2/PR05/home/sub-PR05_stage-2_audio-athome_signal-preproc_wiener_filtering'

    loudness_csv_output_directory = '/userdata/msharma/sub-PR05-stage-2_audio-audiotype_preproc_wiener_filtering_metadata_and_plots/sub-PR05_stage-2_audio-audiotype_preproc_wiener_filtering_loudness_metadata'
    loudness_plot_output_directory = '/userdata/msharma/sub-PR05-stage-2_audio-audiotype_preproc_wiener_filtering_metadata_and_plots/sub-PR05_stage-2_audio-audiotype_preproc_wiener_filtering_loudness_plots'

    os.makedirs(loudness_csv_output_directory, exist_ok=True)
    os.makedirs(loudness_plot_output_directory, exist_ok=True)

    # Collect per-recording metrics for the summary plots
    recording_labels = []
    mean_intensities = []
    intensity_stds = []
    intensity_medians = []
    intensity_q1s = []
    intensity_q3s = []
    intensity_iqrs = []

    for num in range(579, 1067):
        audio_name_without_wav = str(num)
        audio_name = 'sub-PR05_stage-2_audio-athome_signal-preproc_wiener_' + audio_name_without_wav + '.wav'
        sound_path = os.path.join(patient_preproc_data_directory, audio_name)

        if not os.path.exists(sound_path):
            continue

        (ts, intensity_db, mean_intensity_db, sampling_hz,
         i_std, i_median, i_iqr, i_q1, i_q3) = loudness_in_db(
            sound_path,
            loudness_csv_output_directory,
        )

        plot_path = os.path.join(
            loudness_plot_output_directory,
            'sub-PR05_stage-2_audio-audiotype_preproc_wiener_' + audio_name_without_wav + '_loudness.png'
        )
        save_loudness_plot(ts, intensity_db, mean_intensity_db, plot_path)

        # Store for summary
        recording_labels.append(audio_name_without_wav)
        mean_intensities.append(mean_intensity_db)
        intensity_stds.append(i_std)
        intensity_medians.append(i_median)
        intensity_q1s.append(i_q1)
        intensity_q3s.append(i_q3)
        intensity_iqrs.append(i_iqr)

    # ── Simple point plots (original + new metrics) ──
    save_summary_point_plot(
        recording_labels, mean_intensities,
        ylabel="Mean Intensity (dB)",
        title="Mean Intensity (dB) Across Recordings",
        output_path=os.path.join(loudness_plot_output_directory,
                                 'sub-PR05_stage-2_wiener_loudness_mean_summary.png'),
    )

    save_summary_point_plot(
        recording_labels, intensity_stds,
        ylabel="Intensity Std Dev (dB)",
        title="Intensity Standard Deviation Across Recordings",
        output_path=os.path.join(loudness_plot_output_directory,
                                 'sub-PR05_stage-2_wiener_loudness_std_summary.png'),
    )

    save_summary_point_plot(
        recording_labels, intensity_iqrs,
        ylabel="Intensity IQR (dB)",
        title="Intensity Interquartile Range Across Recordings",
        output_path=os.path.join(loudness_plot_output_directory,
                                 'sub-PR05_stage-2_wiener_loudness_iqr_summary.png'),
    )

    # ── Error-bar plots (glacier-style) ──

    # Mean ± 1 SD
    stds_arr = np.array(intensity_stds, dtype=float)
    save_summary_errorbar_plot(
        recording_labels,
        centers=mean_intensities,
        lower_bars=stds_arr.tolist(),
        upper_bars=stds_arr.tolist(),
        ylabel="Intensity (dB)",
        title="Mean Intensity ± 1 SD Across Recordings",
        output_path=os.path.join(loudness_plot_output_directory,
                                 'sub-PR05_stage-2_wiener_loudness_mean_sd_errorbar.png'),
        bar_label="Mean ± 1 SD",
    )

    # Median with IQR bars (Q1 to Q3)
    medians_arr = np.array(intensity_medians, dtype=float)
    q1_arr = np.array(intensity_q1s, dtype=float)
    q3_arr = np.array(intensity_q3s, dtype=float)
    save_summary_errorbar_plot(
        recording_labels,
        centers=intensity_medians,
        lower_bars=(medians_arr - q1_arr).tolist(),
        upper_bars=(q3_arr - medians_arr).tolist(),
        ylabel="Intensity (dB)",
        title="Median Intensity with IQR Across Recordings",
        output_path=os.path.join(loudness_plot_output_directory,
                                 'sub-PR05_stage-2_wiener_loudness_median_iqr_errorbar.png'),
        bar_label="Median [Q1, Q3]",
    )


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