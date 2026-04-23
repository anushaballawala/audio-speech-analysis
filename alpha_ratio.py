import parselmouth
import numpy as np
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
import time
import os
import csv

# # snd = "raw_audio/hoarse_test_voice.wav"
# snd = "raw_audio/testsoundmono.mp3" # gives mean alpha ratio 2.8307666078269107 w/ log10
snd = "raw_audio/high_pitch.wav" # gives mean alpha ratio of 1.4452959901226703 w/ log10


def alpha_ratio( #TODO ask if I should add masking to alpha_ratio
    sound_path: str,
    csv_folder_name: str,
    time_step: float = 0.01, 
    window_length: float = 0.025,
    f_low1: float = 50.0,
    f_high1: float = 1000.0,
    f_low2: float = 1000.0,
    f_high2: float = 5000.0,
    pitch_floor: float = 115.0,
    pitch_ceiling: float = 400.0,
    stats: bool = True
):
    """
    Linear alpha ratio per frame:
         alpha(t) = log10(sum(E[50-1000 Hz]) / sum(E[1000-5000 Hz]))
         
    Args:
    sound_path: path to .csv sound file
    csv_folder_name (str): name of folder where stats csv file should be placed (doesn't matter if stats is False).
    stats (bool): Whether the function should output a csv including time taken to execute and other metadata.
    
    Returns:
      speaking_times: np.ndarray (s)
      alpha_ratios: np.ndarray
      alpha_ratio_mean: float
      sampling_hz: float
      alpha_ratio_std: float          — std dev of alpha ratio values
      alpha_ratio_median: float       — median of alpha ratio values
      alpha_ratio_iqr: float          — interquartile range (Q3 - Q1)
      alpha_ratio_q1: float           — 25th percentile
      alpha_ratio_q3: float           — 75th percentile
    """
    if(stats):
        start_time = time.perf_counter()
        wav_base = os.path.splitext(os.path.basename(sound_path))[0]
        func_name = alpha_ratio.__name__
        stats_csv_file_name = f"{csv_folder_name}/{wav_base}_{func_name}.csv"
    
    #NEW: FOR MASKING USING F0 (PITCH):
    # pitch = sound.to_pitch(time_step=time_step, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    
    sound = parselmouth.Sound(sound_path)
    sampling_hz = sound.sampling_frequency

    pitch = sound.to_pitch(time_step=time_step, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    
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
    
    low_freq_summed_power = np.sum(low_freq_PSDs, axis=0)  # sum of PSD over low band per frame
    high_freq_summed_power = np.sum(high_freq_PSDs, axis=0)  # sum of PSD over high band per frame

    alpha_ratios = []
    speaking_times = []
    for j, t in enumerate(times):
        f0 = pitch.get_value_at_time(float(t))
        if np.isnan(f0):
            continue  # person is not speaking
        speaking_times.append(t)

        # Avoid divide-by-zero; skip frames where high-band energy is 0 or non-finite.
        denom = high_freq_summed_power[j]
        num = low_freq_summed_power[j]
        if (not np.isfinite(num)) or (not np.isfinite(denom)) or denom <= 0 or num <= 0:
            continue

        alpha_ratios.append(np.log10(num / denom))

    alpha_ratios = np.array(alpha_ratios)
    alpha_ratio_mean = float(np.mean(alpha_ratios)) if alpha_ratios.size else float("nan")
    speaking_times = np.array(speaking_times)

    # ── New spread / dispersion metrics ──
    if alpha_ratios.size > 0:
        alpha_ratio_std = float(np.std(alpha_ratios, ddof=1)) if alpha_ratios.size > 1 else 0.0
        alpha_ratio_median = float(np.median(alpha_ratios))
        alpha_ratio_q1 = float(np.percentile(alpha_ratios, 25))
        alpha_ratio_q3 = float(np.percentile(alpha_ratios, 75))
        alpha_ratio_iqr = alpha_ratio_q3 - alpha_ratio_q1
    else:
        alpha_ratio_std = float("nan")
        alpha_ratio_median = float("nan")
        alpha_ratio_q1 = float("nan")
        alpha_ratio_q3 = float("nan")
        alpha_ratio_iqr = float("nan")
    
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
                "window_length",
                "f_low1",
                "f_high1",
                "f_low2",
                "f_high2",
                "pitch_floor",
                "pitch_ceiling",
                "alpha_ratio_mean",
                "alpha_ratio_std",
                "alpha_ratio_median",
                "alpha_ratio_q1",
                "alpha_ratio_q3",
                "alpha_ratio_iqr",
                "elapsed_seconds",
                "time_seconds",
                "alpha_ratio",
            ])
            if alpha_ratios.size == 0:
                writer.writerow([
                    os.path.basename(sound_path),
                    sampling_hz,
                    time_step,
                    window_length,
                    f_low1,
                    f_high1,
                    f_low2,
                    f_high2,
                    pitch_floor,
                    pitch_ceiling,
                    fmt(alpha_ratio_mean),
                    fmt(alpha_ratio_std),
                    fmt(alpha_ratio_median),
                    fmt(alpha_ratio_q1),
                    fmt(alpha_ratio_q3),
                    fmt(alpha_ratio_iqr),
                    f"{elapsed_sec:.6f}",
                    "",
                    "",
                ])
            else:
                for i, (t, a) in enumerate(zip(speaking_times, alpha_ratios)):
                    writer.writerow([
                        os.path.basename(sound_path) if i == 0 else "",
                        sampling_hz if i == 0 else "",
                        time_step if i == 0 else "",
                        window_length if i == 0 else "",
                        f_low1 if i == 0 else "",
                        f_high1 if i == 0 else "",
                        f_low2 if i == 0 else "",
                        f_high2 if i == 0 else "",
                        pitch_floor if i == 0 else "",
                        pitch_ceiling if i == 0 else "",
                        fmt(alpha_ratio_mean) if i == 0 else "",
                        fmt(alpha_ratio_std) if i == 0 else "",
                        fmt(alpha_ratio_median) if i == 0 else "",
                        fmt(alpha_ratio_q1) if i == 0 else "",
                        fmt(alpha_ratio_q3) if i == 0 else "",
                        fmt(alpha_ratio_iqr) if i == 0 else "",
                        f"{elapsed_sec:.6f}" if i == 0 else "",
                        f"{float(t):.6f}",
                        f"{float(a):.6f}",
                    ])
    return (speaking_times, alpha_ratios, alpha_ratio_mean, sampling_hz,
            alpha_ratio_std, alpha_ratio_median, alpha_ratio_iqr, alpha_ratio_q1, alpha_ratio_q3)


def save_alpha_ratio_plot(
    ts: np.ndarray,
    ratios: np.ndarray,
    alpha_ratio_mean: float,
    output_path: str,
):
    """Save alpha ratio plot in the same format as the TESTING section."""
    if ts.size == 0 or ratios.size == 0:
        return

    plt.figure()
    plt.scatter(ts, ratios, color='blue', label='Data Points')
    plt.axhline(y=alpha_ratio_mean, color='r', linestyle='-',
                label=f'Alpha ratio mean: {alpha_ratio_mean}')
    plt.title("Alpha ratios over time")
    plt.xlabel("Time (in seconds)")
    plt.ylabel("Alpha ratio")
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
    """
    if len(values) == 0:
        return

    labels = np.array(recording_labels)
    vals = np.array(values, dtype=float)

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
        yerr=np.array([lo, hi]),
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
    patient_preproc_data_directory = '/data_store2/resection/neuropsych_video/presidio/Stage2/PR09/home/sub-PR09_stage-2_audio-athome_signal-preproc_spectral_gating_100_percent'

    alpha_ratio_csv_output_directory = '/userdata/msharma/sub-PR09-stage-2_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots/sub-PR09_stage-2_audio-audiotype_preproc_spectral_gating_100_percent_alpha_ratio_metadata'
    alpha_ratio_plot_output_directory = '/userdata/msharma/sub-PR09-stage-2_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots/sub-PR09_stage-2_audio-audiotype_preproc_spectral_gating_100_percent_alpha_ratio_plots'

    os.makedirs(alpha_ratio_csv_output_directory, exist_ok=True)
    os.makedirs(alpha_ratio_plot_output_directory, exist_ok=True)

    # Collect per-recording metrics for the summary plots
    recording_labels = []
    mean_alpha_ratios = []
    alpha_ratio_stds = []
    alpha_ratio_medians = []
    alpha_ratio_q1s = []
    alpha_ratio_q3s = []
    alpha_ratio_iqrs = []

    for num in range(1, 122):
        audio_name_without_wav = str(num)
        audio_name = 'sub-PR09_stage-2_audio-athome_signal-preproc_' + audio_name_without_wav + '.wav'
        sound_path = os.path.join(patient_preproc_data_directory, audio_name)

        if not os.path.exists(sound_path):
            continue

        (ts, ratios, alpha_ratio_mean, s_hz,
         ar_std, ar_median, ar_iqr, ar_q1, ar_q3) = alpha_ratio(
            sound_path,
            alpha_ratio_csv_output_directory,
        )

        plot_path = os.path.join(
            alpha_ratio_plot_output_directory,
            'sub-PR09_stage-2_audio-audiotype_preproc_' + audio_name_without_wav + '_alpha_ratio.png'
        )
        save_alpha_ratio_plot(ts, ratios, alpha_ratio_mean, plot_path)

        # Store for summary
        recording_labels.append(audio_name_without_wav)
        mean_alpha_ratios.append(alpha_ratio_mean)
        alpha_ratio_stds.append(ar_std)
        alpha_ratio_medians.append(ar_median)
        alpha_ratio_q1s.append(ar_q1)
        alpha_ratio_q3s.append(ar_q3)
        alpha_ratio_iqrs.append(ar_iqr)

    # ── Simple point plots (original + new metrics) ──
    save_summary_point_plot(
        recording_labels, mean_alpha_ratios,
        ylabel="Mean Alpha Ratio",
        title="Mean Alpha Ratio Across Recordings",
        output_path=os.path.join(alpha_ratio_plot_output_directory,
                                 'sub-PR09_stage-2_alpha_ratio_mean_summary.png'),
    )

    save_summary_point_plot(
        recording_labels, alpha_ratio_stds,
        ylabel="Alpha Ratio Std Dev",
        title="Alpha Ratio Standard Deviation Across Recordings",
        output_path=os.path.join(alpha_ratio_plot_output_directory,
                                 'sub-PR09_stage-2_alpha_ratio_std_summary.png'),
    )

    save_summary_point_plot(
        recording_labels, alpha_ratio_iqrs,
        ylabel="Alpha Ratio IQR",
        title="Alpha Ratio Interquartile Range Across Recordings",
        output_path=os.path.join(alpha_ratio_plot_output_directory,
                                 'sub-PR09_stage-2_alpha_ratio_iqr_summary.png'),
    )

    # ── Error-bar plots (glacier-style) ──

    # Mean ± 1 SD
    stds_arr = np.array(alpha_ratio_stds, dtype=float)
    save_summary_errorbar_plot(
        recording_labels,
        centers=mean_alpha_ratios,
        lower_bars=stds_arr.tolist(),
        upper_bars=stds_arr.tolist(),
        ylabel="Alpha Ratio",
        title="Mean Alpha Ratio ± 1 SD Across Recordings",
        output_path=os.path.join(alpha_ratio_plot_output_directory,
                                 'sub-PR09_stage-2_alpha_ratio_mean_sd_errorbar.png'),
        bar_label="Mean ± 1 SD",
    )

    # Median with IQR bars (Q1 to Q3)
    medians_arr = np.array(alpha_ratio_medians, dtype=float)
    q1_arr = np.array(alpha_ratio_q1s, dtype=float)
    q3_arr = np.array(alpha_ratio_q3s, dtype=float)
    save_summary_errorbar_plot(
        recording_labels,
        centers=alpha_ratio_medians,
        lower_bars=(medians_arr - q1_arr).tolist(),
        upper_bars=(q3_arr - medians_arr).tolist(),
        ylabel="Alpha Ratio",
        title="Median Alpha Ratio with IQR Across Recordings",
        output_path=os.path.join(alpha_ratio_plot_output_directory,
                                 'sub-PR09_stage-2_alpha_ratio_median_iqr_errorbar.png'),
        bar_label="Median [Q1, Q3]",
    )


if __name__ == "__main__":
    main()


#__________TESTING___________

# print(alpha_ratio_mean)
# plt.scatter(ts, ratios, color='blue', label='Data Points')
# plt.axhline(y=alpha_ratio_mean, color='r', linestyle='-', label=f'Alpha ratio mean: {alpha_ratio_mean}')
# plt.title("Alpha ratios over time")
# plt.xlabel("Time (in seconds)")
# plt.ylabel("Alpha ratio")
# # plt.savefig("figures/alpha_ratios_high_pitched_audio.png", dpi=300, bbox_inches='tight')
# plt.show()