import numpy as np
import parselmouth
import scipy
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

# demean_wav_sound(snd, "processed_audio/wavtestsoundmono_demeaned.wav")

# def butterworth_highpass_filter(data, srate, cutoff=80, order=5):

#     # 1. Normalize cutoff by Nyquist frequency
#     # (High-pass only needs one value, not a range)
#     nyq = 0.5 * srate
#     normal_cutoff = cutoff / nyq
    
#     # 2. Change btype to 'high'
#     sos = butter(order, normal_cutoff, btype='high', output='sos')

#     # 3. Apply filter
#     if data.ndim == 1:
#         return sosfiltfilt(sos, data)
    
#     filt_data = np.zeros_like(data)
#     for iChan in range(data.shape[1]):
#         filt_data[:, iChan] = sosfiltfilt(sos, data[:, iChan])
        
#     return filt_data