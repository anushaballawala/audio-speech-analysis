# apqN shimmer is N-point period amplitude perturbation quotient (apq5 is supposed to be most effective)

import numpy as np
import parselmouth
from parselmouth.praat import call
import matplotlib.pyplot as plt
import time
import os
import glob
import csv


def shimmer_apqN(
    sound_path: str,
    csv_folder_name: str,
    N: int,  # N can only be 3, 5, or 11
    *,
    pitch_floor: float = 75.0,
    pitch_ceiling: float = 500.0,
    pitch_time_step=0.01,
    from_time: float = 0.0,  # if from_time and to_time are same it goes for the entire audio recording
    to_time: float = 0.0,
    period_floor: float = 0.0001,  # all of these are default praat values
    period_ceiling: float = 0.02,
    maximum_period_factor: float = 1.3,
    maximum_amplitude_factor: float = 1.6,
    stats: bool = True,
):
    """
    Compute N-point Amplitude Perturbation Quotient (apqN shimmer).
    """

    sound = parselmouth.Sound(sound_path)
    sampling_hz = sound.sampling_frequency

    if stats:
        start_time = time.perf_counter()
        wav_base = os.path.splitext(os.path.basename(sound_path))[0]
        func_name = shimmer_apqN.__name__
        stats_csv_file_name = f"{csv_folder_name}/{wav_base}_{func_name}.csv"

    if N not in [3, 5, 11]:
        raise ValueError("N can only be 3, 5, or 11")

    pitch = sound.to_pitch(time_step=pitch_time_step, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)

    point = call([sound, pitch], "To PointProcess (cc)")

    apqN = call(
        [sound, point],
        f"Get shimmer (apq{N})",
        from_time,
        to_time,
        period_floor,
        period_ceiling,
        maximum_period_factor,
        maximum_amplitude_factor,
    )

    if stats:
        elapsed_sec = time.perf_counter() - start_time
        with open(stats_csv_file_name, "x", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "sound_path",
                "sample_rate_hz",
                "N",
                "pitch_floor",
                "pitch_ceiling",
                "pitch_time_step",
                "from_time",
                "to_time",
                "period_floor",
                "period_ceiling",
                "maximum_period_factor",
                "maximum_amplitude_factor",
                "shimmer_val",
                "elapsed_seconds",
            ])
            writer.writerow([
                os.path.basename(sound_path),
                sampling_hz,
                N,
                pitch_floor,
                pitch_ceiling,
                pitch_time_step,
                from_time,
                to_time,
                period_floor,
                period_ceiling,
                maximum_period_factor,
                maximum_amplitude_factor,
                apqN,
                f"{elapsed_sec:.6f}",
            ])

    return apqN, sampling_hz


def save_summary_point_plot(
    recording_labels: list,
    values: list,
    ylabel: str,
    title: str,
    output_path: str,
):
    """Scatter of one shimmer value per recording, with grand-mean reference line."""
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
               label=f'Grand mean: {grand_mean:.6f}')

    n = len(labels)
    if n > 40:
        tick_step = max(1, n // 20)
        tick_indices = list(range(0, n, tick_step))
        ax.set_xticks(tick_indices)
        ax.set_xticklabels(labels[tick_indices], rotation=90)
    else:
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=90)

    ax.set_xlabel("Recording")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    patient_preproc_data_directory = '/data_store2/resection/neuropsych_video/presidio/Stage2/ClinicianScales/PR05/PR05_clinician_scales_audio_preproc_spectral_gating_100_percent'

    shimmer_csv_output_directory = '/userdata/msharma/sub-PR05-clinician_scales_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots/sub-PR05_clinician_scales_audio-audiotype_preproc_spectral_gating_100_percent_shimmer_metadata'
    shimmer_plot_output_directory = '/userdata/msharma/sub-PR05-clinician_scales_audio-audiotype_preproc_spectral_gating_100_percent_metadata_and_plots/sub-PR05_clinician_scales_audio-audiotype_preproc_spectral_gating_100_percent_shimmer_plots'

    os.makedirs(shimmer_csv_output_directory, exist_ok=True)
    os.makedirs(shimmer_plot_output_directory, exist_ok=True)

    recording_labels = []
    shimmer_vals = []

    for sound_path in sorted(glob.glob(os.path.join(patient_preproc_data_directory, '*.wav'))):
        audio_name_without_wav = os.path.splitext(os.path.basename(sound_path))[0]

        apq5_val, _ = shimmer_apqN(sound_path, shimmer_csv_output_directory, 5)
        recording_labels.append(audio_name_without_wav)
        shimmer_vals.append(apq5_val)

    save_summary_point_plot(
        recording_labels, shimmer_vals,
        ylabel="Shimmer (apq5)",
        title="Shimmer (apq5) Across Recordings",
        output_path=os.path.join(shimmer_plot_output_directory,
                                 'sub-PR05_clinician_scales_shimmer_apq5_summary.png'),
    )


if __name__ == "__main__":
    main()
