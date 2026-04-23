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
    pitch_floor: float = 115.0,
    pitch_ceiling: float = 400.0,
    stats: bool = True
):
    """
    Returns:
      nonzero_xs: np.ndarray (s)
      nonzero_f0_values: np.ndarray
      f0_lstsq_slope: float
      f0_lstsq_intercept: float
      pitch_mean: float
      sampling_hz: float
      pitch_std: float          — std dev of nonzero f0 values
      pitch_median: float       — median of nonzero f0 values
      pitch_iqr: float          — interquartile range (Q3 - Q1)
      pitch_q1: float           — 25th percentile
      pitch_q3: float           — 75th percentile
      lstsq_residual_std: float — std dev of residuals from the least-squares line
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

    # ── New spread / dispersion metrics ──
    if nonzero_f0_values.size > 0:
        pitch_std = float(np.std(nonzero_f0_values, ddof=1)) if nonzero_f0_values.size > 1 else 0.0
        pitch_median = float(np.median(nonzero_f0_values))
        pitch_q1 = float(np.percentile(nonzero_f0_values, 25))
        pitch_q3 = float(np.percentile(nonzero_f0_values, 75))
        pitch_iqr = pitch_q3 - pitch_q1
    else:
        pitch_std = float("nan")
        pitch_median = float("nan")
        pitch_q1 = float("nan")
        pitch_q3 = float("nan")
        pitch_iqr = float("nan")

    # least squares Ax = b
    A = np.column_stack((nonzero_xs, np.ones(len(nonzero_xs))))
    b = nonzero_f0_values

    lstsqsoln = lstsq(A, b, rcond=None)[0]

    f0_lstsq_slope = lstsqsoln[0] #measure referenced in paper as a strong feature
    f0_lstsq_intercept = lstsqsoln[1]

    # Residual std: how much individual f0 values scatter around the trend line
    if nonzero_f0_values.size > 2:
        residuals = nonzero_f0_values - (f0_lstsq_slope * nonzero_xs + f0_lstsq_intercept)
        lstsq_residual_std = float(np.std(residuals, ddof=2))  # ddof=2 because 2 params estimated
    else:
        lstsq_residual_std = float("nan")
    
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
                "pitch_std",
                "pitch_median",
                "pitch_q1",
                "pitch_q3",
                "pitch_iqr",
                "lstsq_residual_std",
                "elapsed_seconds",
                "time_seconds",
                "f0_hz",
            ])
            fmt = lambda v: f"{v:.6f}" if np.isfinite(v) else ""
            if nonzero_xs.size == 0:
                writer.writerow([
                    os.path.basename(sound_path),
                    sampling_hz,
                    time_step,
                    pitch_floor,
                    pitch_ceiling,
                    f0_lstsq_slope,
                    fmt(pitch_mean),
                    fmt(pitch_std),
                    fmt(pitch_median),
                    fmt(pitch_q1),
                    fmt(pitch_q3),
                    fmt(pitch_iqr),
                    fmt(lstsq_residual_std),
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
                        fmt(pitch_mean) if i == 0 else "",
                        fmt(pitch_std) if i == 0 else "",
                        fmt(pitch_median) if i == 0 else "",
                        fmt(pitch_q1) if i == 0 else "",
                        fmt(pitch_q3) if i == 0 else "",
                        fmt(pitch_iqr) if i == 0 else "",
                        fmt(lstsq_residual_std) if i == 0 else "",
                        f"{elapsed_sec:.6f}" if i == 0 else "",
                        f"{t:.6f}",
                        f"{f0:.3f}",
                    ])
    
    return (nonzero_xs, nonzero_f0_values, f0_lstsq_slope, f0_lstsq_intercept,
            pitch_mean, sampling_hz,
            pitch_std, pitch_median, pitch_iqr, pitch_q1, pitch_q3, lstsq_residual_std)


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
    Save a summary plot with error bars (like the glacier Δh plot).

    Each recording gets a point at `centers[i]` with asymmetric error bars
    extending down by `lower_bars[i]` and up by `upper_bars[i]`.

    Args:
        recording_labels: list of recording identifiers.
        centers: central value per recording (e.g. mean pitch).
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


 # nonzero_xs, nonzero_f0_values, f0_lstsq_slope, f0_lstsq_intercept, s_hz = pitches(snd, 'function_output_data')

def main():
    # Input directory and naming convention must match the provided patient script
    patient_preproc_data_directory = '/data_store2/resection/neuropsych_video/presidio/Stage2/PR05/home/sub-PR05_stage-2_audio-athome_signal-preproc_wiener_filtering'

    # Output directories (as requested)
    pitch_csv_output_directory = '/userdata/msharma/sub-PR05-stage-2_audio-audiotype_preproc_wiener_filtering_metadata_and_plots/sub-PR05_stage-2_audio-audiotype_preproc_wiener_filtering_pitch_metadata'
    pitch_plot_output_directory = '/userdata/msharma/sub-PR05-stage-2_audio-audiotype_preproc_wiener_filtering_metadata_and_plots/sub-PR05_stage-2_audio-audiotype_preproc_wiener_filtering_pitch_plots'

    os.makedirs(pitch_csv_output_directory, exist_ok=True)
    os.makedirs(pitch_plot_output_directory, exist_ok=True)

    # Collect per-recording metrics for the summary plots
    recording_labels = []
    pitch_means = []
    pitch_stds = []
    pitch_medians = []
    pitch_q1s = []
    pitch_q3s = []
    pitch_iqrs = []
    f0_slopes = []
    f0_intercepts = []
    lstsq_residual_stds = []

    for num in range(579, 1067):
        audio_name_without_wav = str(num)
        audio_name = 'sub-PR05_stage-2_audio-athome_signal-preproc_wiener_' + audio_name_without_wav + '.wav'
        sound_path = os.path.join(patient_preproc_data_directory, audio_name)
        
        if not os.path.exists(sound_path):
            continue

        (nonzero_xs, nonzero_f0_values, f0_lstsq_slope, f0_lstsq_intercept,
         pitch_mean, s_hz,
         pitch_std, pitch_median, pitch_iqr, pitch_q1, pitch_q3,
         lstsq_residual_std) = pitches(
            sound_path,
            pitch_csv_output_directory,
        )

        plot_path = os.path.join(
            pitch_plot_output_directory,
            'sub-PR05_stage-2_audio-audiotype_preproc_wiener_' + audio_name_without_wav + '_pitch.png'
        )
        save_pitch_plot(nonzero_xs, nonzero_f0_values, f0_lstsq_slope, f0_lstsq_intercept, plot_path)

        # Store for summary
        recording_labels.append(audio_name_without_wav)
        pitch_means.append(pitch_mean)
        pitch_stds.append(pitch_std)
        pitch_medians.append(pitch_median)
        pitch_q1s.append(pitch_q1)
        pitch_q3s.append(pitch_q3)
        pitch_iqrs.append(pitch_iqr)
        f0_slopes.append(f0_lstsq_slope)
        f0_intercepts.append(f0_lstsq_intercept)
        lstsq_residual_stds.append(lstsq_residual_std)

    # ── Simple point plots (original style) ──
    save_summary_point_plot(
        recording_labels, pitch_means,
        ylabel="Mean Pitch (Hz)",
        title="Mean Pitch Across Recordings",
        output_path=os.path.join(pitch_plot_output_directory,
                                 'sub-PR05_stage-2_wiener_pitch_mean_summary.png'),
    )

    save_summary_point_plot(
        recording_labels, f0_slopes,
        ylabel="F0 Least-Squares Slope",
        title="F0 Least-Squares Slope Across Recordings",
        output_path=os.path.join(pitch_plot_output_directory,
                                 'sub-PR05_stage-2_wiener_f0_lstsq_slope_summary.png'),
    )

    save_summary_point_plot(
        recording_labels, f0_intercepts,
        ylabel="F0 Least-Squares Intercept (Hz)",
        title="F0 Least-Squares Intercept Across Recordings",
        output_path=os.path.join(pitch_plot_output_directory,
                                 'sub-PR05_stage-2_wiener_f0_lstsq_intercept_summary.png'),
    )

    # ── New: point plot for std dev itself ──
    save_summary_point_plot(
        recording_labels, pitch_stds,
        ylabel="Pitch Std Dev (Hz)",
        title="Pitch Standard Deviation Across Recordings",
        output_path=os.path.join(pitch_plot_output_directory,
                                 'sub-PR05_stage-2_wiener_pitch_std_summary.png'),
    )

    save_summary_point_plot(
        recording_labels, pitch_iqrs,
        ylabel="Pitch IQR (Hz)",
        title="Pitch Interquartile Range Across Recordings",
        output_path=os.path.join(pitch_plot_output_directory,
                                 'sub-PR05_stage-2_wiener_pitch_iqr_summary.png'),
    )

    # ── New: error-bar plots (glacier-style) ──

    # Mean ± 1 SD
    means_arr = np.array(pitch_means, dtype=float)
    stds_arr = np.array(pitch_stds, dtype=float)
    save_summary_errorbar_plot(
        recording_labels,
        centers=pitch_means,
        lower_bars=stds_arr.tolist(),   # mean - 1 SD
        upper_bars=stds_arr.tolist(),   # mean + 1 SD
        ylabel="Pitch (Hz)",
        title="Mean Pitch ± 1 SD Across Recordings",
        output_path=os.path.join(pitch_plot_output_directory,
                                 'sub-PR05_stage-2_wiener_pitch_mean_sd_errorbar.png'),
        bar_label="Mean ± 1 SD",
    )

    # Median with IQR bars (Q1 to Q3)
    medians_arr = np.array(pitch_medians, dtype=float)
    q1_arr = np.array(pitch_q1s, dtype=float)
    q3_arr = np.array(pitch_q3s, dtype=float)
    save_summary_errorbar_plot(
        recording_labels,
        centers=pitch_medians,
        lower_bars=(medians_arr - q1_arr).tolist(),  # median − Q1
        upper_bars=(q3_arr - medians_arr).tolist(),  # Q3 − median
        ylabel="Pitch (Hz)",
        title="Median Pitch with IQR Across Recordings",
        output_path=os.path.join(pitch_plot_output_directory,
                                 'sub-PR05_stage-2_wiener_pitch_median_iqr_errorbar.png'),
        bar_label="Median [Q1, Q3]",
    )

    # Slope ± residual std (shows how noisy each recording's trend is)
    save_summary_errorbar_plot(
        recording_labels,
        centers=f0_slopes,
        lower_bars=lstsq_residual_stds,
        upper_bars=lstsq_residual_stds,
        ylabel="F0 Slope (Hz/s)",
        title="F0 Slope ± Residual Std Across Recordings",
        output_path=os.path.join(pitch_plot_output_directory,
                                 'sub-PR05_stage-2_wiener_f0_slope_residual_errorbar.png'),
        bar_label="Slope ± residual SD",
    )


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