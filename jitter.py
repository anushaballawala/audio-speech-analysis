import numpy as np
import parselmouth
from parselmouth.praat import call
import matplotlib.pyplot as plt

# snd = parselmouth.Sound("raw_audio/hoarse_test_voice.wav") # jitter value: 0.07892894876979728 (higher, as expected)
snd = parselmouth.Sound("raw_audio/testsoundmono.mp3") # jitter value: 0.02721768951093052
# snd = parselmouth.Sound("raw_audio/high_pitch.wav")


def jitter(
    sound: parselmouth.Sound,
    kind : str = "local",
    pitch_floor: float = 75.0, #currently set to 75Hz, seems to work well
    pitch_ceiling: float = 500.0,
    pitch_time_step = 0.01,
    from_time: float = 0.0, # if from time and to time are same it goes for the entire audio recording
    to_time: float = 0.0,
    period_floor: float = 0.0001, # this and below are default praat vals
    period_ceiling: float = 0.02,
    maximum_period_factor: float = 1.3,
):
    """
    Returns jitter of sound.
    
    kind options: "local", "local, absolute", "rap", "ppq5", "ddp"
    """

    if kind not in ["local", "local, absolute", "rap", "ppq5", "ddp"]:
        raise ValueError("Kind option not one of those allowed. Look at docstring for kind options.")
    
    pitch = sound.to_pitch(time_step=pitch_time_step, pitch_floor=pitch_floor, pitch_ceiling=pitch_ceiling)
    
    point = call([sound, pitch], "To PointProcess (cc)")

    jtr = call(
        point,
        f"Get jitter ({kind})",
        from_time,
        to_time,
        period_floor,
        period_ceiling,
        maximum_period_factor,
    )
    
    return jtr

jtr_val = jitter(snd)

print(jtr_val)