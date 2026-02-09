import scipy.io.wavfile as wavfile
import numpy as np
import time
import os
import csv


sample_rate_buttered, data_buttered = wavfile.read('processed_audio/wavtestsoundmono_butterworth.wav') # max abs value: 3724, min val: -2599
sample_rate, data = wavfile.read('raw_audio/wavtestsoundmono.wav') # max abs value: 3700, min abs val: -2593

if len(data.shape) > 1:
    data = data.flatten()

max_val = np.max(np.abs(data))
min_val = np.min(data)

if len(data_buttered.shape) > 1:
    data_buttered = data_buttered.flatten()

max_val_buttered = np.max(np.abs(data_buttered))
min_val_buttered = np.min(data_buttered)


print(f"Maximum absolute value: {max_val}")
print(f"Minimum value: {min_val}")
print(f"Maximum absolute value buttered: {max_val_buttered}")
print(f"Minimum value buttered: {min_val_buttered}")