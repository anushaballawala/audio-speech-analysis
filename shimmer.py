# apqN shimmer is N-point period amplitude perturbation quotient (apq5 is supposed to be most effective)

import numpy as np
import parselmouth
from parselmouth.praat import call
import matplotlib.pyplot as plt

snd = parselmouth.Sound("raw_audio/hoarse_test_voice.wav") # apq5 shimmer value: 0.11146423422694232 (higher, as expected)
# snd = parselmouth.Sound("raw_audio/testsoundmono.mp3") # apq5 shimmer value: 0.05805759435795879
# snd = parselmouth.Sound("raw_audio/high_pitch.wav")

def shimmer_apqN(
    sound: parselmouth.Sound,
    N: int,
    *,
    pitch_floor: float = 75.0, #FIXME do lit review to see what correct value of this should be
    pitch_ceiling: float = 500.0,
    pitch_time_step = 0.01,
    from_time: float = 0.0, # if from time and to time are same it goes for the entire audio recording
    to_time: float = 0.0,
    period_floor: float = 0.0001, # all of these are default praat vals
    period_ceiling: float = 0.02,
    maximum_period_factor: float = 1.3,
    maximum_amplitude_factor: float = 1.6,
) -> float:
    """
    Compute N-point Amplitude Perturbation Quotient (aqpN shimmer)
    """
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
    return apqN

apq5_shimmer = shimmer_apqN(snd, 5)
print(apq5_shimmer)
    
    
    