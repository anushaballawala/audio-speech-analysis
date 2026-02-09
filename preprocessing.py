import numpy as np
import parselmouth
import scipy
import time
import os
import csv
from scipy.signal import butter, sosfiltfilt
from scipy.io import wavfile
from parselmouth.praat import call
import matplotlib.pyplot as plt

# snd = "raw_audio/hoarse_test_voice.wav"
snd = "raw_audio/wavtestsoundmono.wav" 
# snd = "raw_audio/high_pitch.wav"



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
            "cutoff_hz"
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

    
    
    

#__________TESTING___________

# demean_wav_sound(snd, "processed_audio/wavtestsoundmono_demeaned.wav")
# wav_butterworth_highpass_filter(snd, "processed_audio/wavtestsoundmono_butterworth.wav")
# demean_and_butterworth_highpass_filter(snd, "processed_audio/wavtestsoundmono_demeaned_and_butterworth.wav", "function_output_data")

    




