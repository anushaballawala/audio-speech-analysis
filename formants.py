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
    pitch_floor: float = 115.0, # PITCH FLOOR AND CEILING ARE FOR DETERMINING FRAMES WHEN A PERSON IS SPEAKING CALCULATED USING NONZERO F0 VALUES. FIXME ASK ABOUT PITCH FLOOR/CELING VALUES TO BE USED; PITCH MIN WAS AN IMPORTANT FEATURE IN DETERMINING DIFFERENCES (w/ 27.5 min pitch was 28.3 With 75 min pitch was around 96.5); 27.5 looks like it creates outliers
    pitch_ceiling: float = 400.0,
    f_i_bandwidth_hz: float = -1,   # get energies +/- f_i_bandwidth_hz/2 Hz around f_i to capture all f_i energy; for defaults, use -1. 
    return_db: bool = True,  # if True, returns 10*log_10(relative_energy)
    stats: bool = True
):
    """
    Returns:
      times: np.ndarray (s)
      rel_energy_f_i: np.ndarray (linear ratio, or dB if return_db=True)
      mean_rel_energy_f_i: float
      sampling_hz: float
      rel_energy_std: float          — std dev of relative energy values
      rel_energy_median: float       — median of relative energy values
      rel_energy_iqr: float          — interquartile range (Q3 - Q1)
      rel_energy_q1: float           — 25th percentile
      rel_energy_q3: float           — 75th percentile
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

    # ── New spread / dispersion metrics ──
    re_arr = np.array(relative_energies, dtype=float)
    if re_arr.size > 0:
        rel_energy_std = float(np.std(re_arr, ddof=1)) if re_arr.size > 1 else 0.0
        rel_energy_median = float(np.median(re_arr))
        rel_energy_q1 = float(np.percentile(re_arr, 25))
        rel_energy_q3 = float(np.percentile(re_arr, 75))
        rel_energy_iqr = rel_energy_q3 - rel_energy_q1
    else:
        rel_energy_std = float("nan")
        rel_energy_median = float("nan")
        rel_energy_q1 = float("nan")
        rel_energy_q3 = float("nan")
        rel_energy_iqr = float("nan")
    
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
                "rel_energy_std",
                "rel_energy_median",
                "rel_energy_q1",
                "rel_energy_q3",
                "rel_energy_iqr",
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
                    fmt(mean_rel_energy_f_i),
                    fmt(rel_energy_std),
                    fmt(rel_energy_median),
                    fmt(rel_energy_q1),
                    fmt(rel_energy_q3),
                    fmt(rel_energy_iqr),
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
                        fmt(mean_rel_energy_f_i) if i == 0 else "",
                        fmt(rel_energy_std) if i == 0 else "",
                        fmt(rel_energy_median) if i == 0 else "",
                        fmt(rel_energy_q1) if i == 0 else "",
                        fmt(rel_energy_q3) if i == 0 else "",
                        fmt(rel_energy_iqr) if i == 0 else "",
                        f"{elapsed_sec:.6f}" if i == 0 else "",
                        f"{float(t):.6f}",
                        f"{float(re):.6f}",
                    ])
    
    return (np.array(speaking_times), np.array(relative_energies), mean_rel_energy_f_i, sampling_hz,
            rel_energy_std, rel_energy_median, rel_energy_iqr, rel_energy_q1, rel_energy_q3)
      
    


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
        centers: central value per recording (e.g. mean relative energy).
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
    patient_preproc_data_directory = '/data_store2/resection/neuropsych_video/presidio/Stage2/PR05/home/sub-PR05_stage-2_audio-athome_signal-preproc'
    
    f3_csv_output_directory = '/userdata/msharma/sub-PR05_stage-2_audio-audiotype_preproc_f3_metadata'
    f3_plot_output_directory = '/userdata/msharma/sub-PR05_stage-2_audio-audiotype_preproc_f3_plots'

    os.makedirs(f3_csv_output_directory, exist_ok=True)
    os.makedirs(f3_plot_output_directory, exist_ok=True)

    # Collect per-recording metrics for the summary plots
    recording_labels = []
    mean_rel_energies_f3 = []
    rel_energy_stds = []
    rel_energy_medians = []
    rel_energy_q1s = []
    rel_energy_q3s = []
    rel_energy_iqrs = []

    for num in range(579, 1067):
        audio_name_without_wav = str(num)
        audio_name = 'sub-PR05_stage-2_audio-athome_signal-preproc_' + audio_name_without_wav + '.wav'
        sound_path = os.path.join(patient_preproc_data_directory, audio_name)

        if not os.path.exists(sound_path):
            continue

        (ts, relative_energies, mean_rel_energy_f_3, sampling_hz,
         re_std, re_median, re_iqr, re_q1, re_q3) = relative_energy_formant(
            sound_path,
            f3_csv_output_directory,
            3,
        )

        plot_path = os.path.join(
            f3_plot_output_directory,
            'sub-PR05_stage-2_audio-audiotype_preproc_' + audio_name_without_wav + '_f3.png'
        )
        save_f3_plot(ts, relative_energies, mean_rel_energy_f_3, plot_path)

        # Store for summary
        recording_labels.append(audio_name_without_wav)
        mean_rel_energies_f3.append(mean_rel_energy_f_3)
        rel_energy_stds.append(re_std)
        rel_energy_medians.append(re_median)
        rel_energy_q1s.append(re_q1)
        rel_energy_q3s.append(re_q3)
        rel_energy_iqrs.append(re_iqr)

    # ── Simple point plots (original + new metrics) ──
    save_summary_point_plot(
        recording_labels, mean_rel_energies_f3,
        ylabel="Mean F3 Relative Energy (dB)",
        title="Mean F3 Relative Energy Across Recordings",
        output_path=os.path.join(f3_plot_output_directory,
                                 'sub-PR05_stage-2_f3_rel_energy_mean_summary.png'),
    )

    save_summary_point_plot(
        recording_labels, rel_energy_stds,
        ylabel="F3 Rel. Energy Std Dev (dB)",
        title="F3 Relative Energy Standard Deviation Across Recordings",
        output_path=os.path.join(f3_plot_output_directory,
                                 'sub-PR05_stage-2_f3_rel_energy_std_summary.png'),
    )

    save_summary_point_plot(
        recording_labels, rel_energy_iqrs,
        ylabel="F3 Rel. Energy IQR (dB)",
        title="F3 Relative Energy Interquartile Range Across Recordings",
        output_path=os.path.join(f3_plot_output_directory,
                                 'sub-PR05_stage-2_f3_rel_energy_iqr_summary.png'),
    )

    # ── Error-bar plots (glacier-style) ──

    # Mean ± 1 SD
    stds_arr = np.array(rel_energy_stds, dtype=float)
    save_summary_errorbar_plot(
        recording_labels,
        centers=mean_rel_energies_f3,
        lower_bars=stds_arr.tolist(),
        upper_bars=stds_arr.tolist(),
        ylabel="F3 Relative Energy (dB)",
        title="Mean F3 Relative Energy ± 1 SD Across Recordings",
        output_path=os.path.join(f3_plot_output_directory,
                                 'sub-PR05_stage-2_f3_rel_energy_mean_sd_errorbar.png'),
        bar_label="Mean ± 1 SD",
    )

    # Median with IQR bars (Q1 to Q3)
    medians_arr = np.array(rel_energy_medians, dtype=float)
    q1_arr = np.array(rel_energy_q1s, dtype=float)
    q3_arr = np.array(rel_energy_q3s, dtype=float)
    save_summary_errorbar_plot(
        recording_labels,
        centers=rel_energy_medians,
        lower_bars=(medians_arr - q1_arr).tolist(),
        upper_bars=(q3_arr - medians_arr).tolist(),
        ylabel="F3 Relative Energy (dB)",
        title="Median F3 Relative Energy with IQR Across Recordings",
        output_path=os.path.join(f3_plot_output_directory,
                                 'sub-PR05_stage-2_f3_rel_energy_median_iqr_errorbar.png'),
        bar_label="Median [Q1, Q3]",
    )


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