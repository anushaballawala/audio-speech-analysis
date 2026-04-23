import numpy as np
import parselmouth
import scipy
import time
import os
import csv
from scipy.signal import butter, sosfiltfilt
from scipy.io import wavfile
from parselmouth.praat import call
# import matplotlib.pyplot as plt

# pip install noisereduce   (required for the denoise function)
import noisereduce as nr


def demean_wav_sound(sound_path, output_path):
    """
    Removes any DC offset from the sound wav file
    """
    sampling_hz, data = wavfile.read(sound_path)
    original_dtype = data.dtype
    data_float = data.astype(np.float64)
    
    mean_val = np.mean(data_float, axis=0)
    demeaned_data = data_float - mean_val
    
    if np.issubdtype(original_dtype, np.integer):
        typ = np.iinfo(original_dtype)
        demeaned_data = np.clip(demeaned_data, typ.min, typ.max)
    
    output_data = demeaned_data.astype(original_dtype)
    
    wavfile.write(output_path, sampling_hz, output_data)
    
    
    
def wav_butterworth_highpass_filter(sound_path, output_path, cutoff=80, order=5):
    """
    Read a WAV file, apply a Butterworth high-pass filter, and write a WAV file.

    Args:
        sound_path (str): input .wav path
        output_path (str): output .wav path
        cutoff (float): cutoff frequency in Hz
        order (int): Butterworth filter order
    """
    sampling_hz, data = wavfile.read(sound_path)

    if cutoff <= 0:
        raise ValueError("cutoff must be > 0")
    nyq = 0.5 * sampling_hz
    if cutoff >= nyq:
        raise ValueError(f"cutoff must be < Nyquist ({nyq} Hz)")

    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype="highpass", output="sos")

    orig_dtype = data.dtype
    is_int = np.issubdtype(orig_dtype, np.integer)

    if is_int:
        max_abs = float(np.iinfo(orig_dtype).max)
        x = data.astype(np.float64) / max_abs
    else:
        x = data.astype(np.float64)

    if x.ndim == 1:
        y = sosfiltfilt(sos, x)
    else:
        y = np.empty_like(x, dtype=np.float64)
        for ch in range(x.shape[1]):
            y[:, ch] = sosfiltfilt(sos, x[:, ch])
            
    y = np.clip(y, -1.0, 1.0)

    if is_int:
        y_out = (y * max_abs).round().astype(orig_dtype)
    else:
        y_out = y.astype(np.float64)

    wavfile.write(output_path, sampling_hz, y_out)
    
    
    
    
def demean_and_butterworth_highpass_filter(
    sound_path: str,
    output_path: str,
    csv_folder_name: str,
    cutoff: float = 80,
    order: int = 5,
    stats: bool = True
):
    """
    Read a WAV file, demean, then apply a Butterworth high-pass filter, and write a WAV file.

    Args:
        sound_path (str): input .wav path
        output_path (str): output .wav path
        csv_folder_name (str): name of folder where stats csv file should be placed (doesn't matter if stats is False).
        cutoff (float): cutoff frequency in Hz
        order (int): Butterworth filter order
        stats (bool): Whether the function should output a csv including time taken to execute and other metadata.
    """
    
    if(stats):
        start_time = time.perf_counter()
        wav_base = os.path.splitext(os.path.basename(sound_path))[0]
        func_name = demean_and_butterworth_highpass_filter.__name__
        stats_csv_file_name = f"{csv_folder_name}/{wav_base}_{func_name}.csv"
    
    # demeaning data
    sampling_hz, data_orig = wavfile.read(sound_path)
    orig_dtype = data_orig.dtype
    
    if cutoff <= 0:
        raise ValueError("cutoff must be > 0")
    nyq = 0.5 * sampling_hz
    if cutoff >= nyq:
        raise ValueError(f"cutoff must be < Nyquist ({nyq} Hz)")
    data_float = data_orig.astype(np.float64)
    
    mean_val = np.mean(data_float, axis=0)
    demeaned_data = data_float - mean_val
    
    # applying butterworth highpass filter
    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype="highpass", output="sos")

    is_int = np.issubdtype(orig_dtype, np.integer)

    if is_int:
        max_abs = float(np.iinfo(orig_dtype).max)
        x = demeaned_data.astype(np.float64) / max_abs
    else:
        x = demeaned_data.astype(np.float64)

    if x.ndim == 1:
        y = sosfiltfilt(sos, x)
    else:
        y = np.empty_like(x, dtype=np.float64)
        for ch in range(x.shape[1]):
            y[:, ch] = sosfiltfilt(sos, x[:, ch])
            
    y = np.clip(y, -1.0, 1.0)

    if is_int:
        y_out = (y * max_abs).round().astype(orig_dtype)
    else:
        y_out = y.astype(np.float64)

    wavfile.write(output_path, sampling_hz, y_out)
    
    if not stats:
        return
    # Timing:
    elapsed_sec = time.perf_counter() - start_time
    with open(stats_csv_file_name, "x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sound_path",
            "output_path",
            "sample_rate_hz",
            "cutoff_hz",
            "order",
            "elapsed_seconds",
        ])
        writer.writerow([
            os.path.basename(sound_path),
            os.path.basename(output_path),
            sampling_hz,
            cutoff,
            order,
            f"{elapsed_sec:.6f}",
        ])


def demean_butterworth_and_denoise(
    sound_path: str,
    output_path: str,
    csv_folder_name: str,
    cutoff: float = 80,
    order: int = 5,
    # ── Spectral-gating denoise parameters ──
    stationary: bool = True,
    prop_decrease: float = 0.99,
    n_fft: int = 2048,
    noise_clip_duration: float = 0.5,
    stats: bool = True,
):
    """
    Read a WAV file, demean, Butterworth high-pass filter, then denoise
    via spectral gating, and write the result.

    Denoising uses the `noisereduce` library (spectral gating).  The noise
    profile is estimated automatically from the quietest portion of the
    recording.  Parameters are set conservatively to avoid musical-noise
    artifacts or speech distortion:

      - stationary=True   : assumes background noise has a roughly constant
                            spectrum (fan, HVAC, hiss). Safer for speech
                            because it won't chase transients. Set False
                            only if the noise itself changes over time.
      - prop_decrease=0.75: reduce detected noise energy by 75 % instead of
                            100 %. Leaving a small residual avoids the
                            "underwater" / phasey artifacts that full
                            subtraction can introduce.
      - n_fft=2048        : FFT size. Larger = finer frequency resolution
                            (better separation of speech harmonics from
                            noise), but coarser time resolution. 2048 @
                            16 kHz ≈ 128 ms windows; good for stationary
                            noise.
      - noise_clip_duration: seconds of the quietest audio used to
                            estimate the noise profile.  0.5 s is usually
                            enough; increase if the recording has long
                            silent gaps with varying noise.

    Args:
        sound_path (str): input .wav path
        output_path (str): output .wav path
        csv_folder_name (str): folder for stats CSV
        cutoff (float): Butterworth high-pass cutoff (Hz)
        order (int): Butterworth filter order
        stationary (bool): assume stationary noise (recommended True)
        prop_decrease (float): 0-1, fraction of noise energy to remove
        n_fft (int): FFT window size for spectral gating
        noise_clip_duration (float): seconds of quietest audio for noise
                                     profile estimation
        stats (bool): write metadata CSV
    """
    if stats:
        start_time = time.perf_counter()
        wav_base = os.path.splitext(os.path.basename(sound_path))[0]
        func_name = demean_butterworth_and_denoise.__name__
        stats_csv_file_name = f"{csv_folder_name}/{wav_base}_{func_name}.csv"

    # ── 1. Read ──
    sampling_hz, data_orig = wavfile.read(sound_path)
    orig_dtype = data_orig.dtype
    is_int = np.issubdtype(orig_dtype, np.integer)

    if cutoff <= 0:
        raise ValueError("cutoff must be > 0")
    nyq = 0.5 * sampling_hz
    if cutoff >= nyq:
        raise ValueError(f"cutoff must be < Nyquist ({nyq} Hz)")

    data_float = data_orig.astype(np.float64)

    # ── 2. Demean ──
    mean_val = np.mean(data_float, axis=0)
    x = data_float - mean_val

    # ── 3. Butterworth high-pass ──
    if is_int:
        max_abs = float(np.iinfo(orig_dtype).max)
        x = x / max_abs

    normal_cutoff = cutoff / nyq
    sos = butter(order, normal_cutoff, btype="highpass", output="sos")

    if x.ndim == 1:
        x = sosfiltfilt(sos, x)
    else:
        for ch in range(x.shape[1]):
            x[:, ch] = sosfiltfilt(sos, x[:, ch])

    x = np.clip(x, -1.0, 1.0)

    # ── 4. Estimate noise profile from the quietest segment ──
    #     RMS energy in short windows → pick the quietest contiguous block.
    noise_samples = int(noise_clip_duration * sampling_hz)
    mono = x if x.ndim == 1 else x[:, 0]

    hop = max(1, noise_samples // 4)
    n_windows = max(1, (len(mono) - noise_samples) // hop + 1)
    rms = np.empty(n_windows)
    for k in range(n_windows):
        s = k * hop
        rms[k] = np.sqrt(np.mean(mono[s : s + noise_samples] ** 2))

    quietest_start = int(np.argmin(rms) * hop)
    quietest_end = min(quietest_start + noise_samples, len(mono))

    if x.ndim == 1:
        noise_clip = x[quietest_start:quietest_end]
    else:
        noise_clip = x[quietest_start:quietest_end, :]

    # ── 5. Spectral-gating denoise ──
    if x.ndim == 1:
        y = nr.reduce_noise(
            y=x,
            sr=sampling_hz,
            y_noise=noise_clip,
            stationary=stationary,
            prop_decrease=prop_decrease,
            n_fft=n_fft,
        )
    else:
        # process each channel independently
        y = np.empty_like(x)
        for ch in range(x.shape[1]):
            y[:, ch] = nr.reduce_noise(
                y=x[:, ch],
                sr=sampling_hz,
                y_noise=noise_clip[:, ch] if noise_clip.ndim > 1 else noise_clip,
                stationary=stationary,
                prop_decrease=prop_decrease,
                n_fft=n_fft,
            )

    y = np.clip(y, -1.0, 1.0)

    # ── 6. Write ──
    if is_int:
        y_out = (y * max_abs).round().astype(orig_dtype)
    else:
        y_out = y.astype(np.float64)

    wavfile.write(output_path, sampling_hz, y_out)

    if not stats:
        return

    elapsed_sec = time.perf_counter() - start_time
    with open(stats_csv_file_name, "x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sound_path",
            "output_path",
            "sample_rate_hz",
            "cutoff_hz",
            "order",
            "stationary",
            "prop_decrease",
            "n_fft",
            "noise_clip_start_s",
            "noise_clip_end_s",
            "elapsed_seconds",
        ])
        writer.writerow([
            os.path.basename(sound_path),
            os.path.basename(output_path),
            sampling_hz,
            cutoff,
            order,
            stationary,
            prop_decrease,
            n_fft,
            f"{quietest_start / sampling_hz:.6f}",
            f"{quietest_end / sampling_hz:.6f}",
            f"{elapsed_sec:.6f}",
        ])

    
if __name__ == "__main__":
    
    patient_raw_data_directory = '/data_store2/resection/neuropsych_video/presidio/Stage2/PR05/home/'
    patient_processed_data_directory = '/data_store2/resection/neuropsych_video/presidio/Stage2/PR05/home/sub-PR05_stage-2_audio-athome_signal-preproc/'
    patient_function_output_directory = '/userdata/msharma/sub-PR05_stage-2_audio-audiotype_preproc_metadata'
    
    for num in range(579, 1067):
        audio_name = str(num) + '_audio.wav'
        audio_name_without_wav = str(num)
        if os.path.exists(patient_raw_data_directory + audio_name):
            demean_butterworth_and_denoise(patient_raw_data_directory + audio_name, patient_processed_data_directory + 'sub-PR05_stage-2_audio-athome_signal-preproc_' + audio_name_without_wav + '.wav', patient_function_output_directory)
    
    

#__________TESTING___________

# demean_wav_sound(snd, "processed_audio/wavtestsoundmono_demeaned.wav")
# wav_butterworth_highpass_filter(snd, "processed_audio/wavtestsoundmono_butterworth.wav")
# demean_and_butterworth_highpass_filter(snd, "processed_audio/wavtestsoundmono_demeaned_and_butterworth.wav", "function_output_data")
# demean_butterworth_and_denoise(snd, "processed_audio/wavtestsoundmono_full_preproc.wav", "function_output_data")