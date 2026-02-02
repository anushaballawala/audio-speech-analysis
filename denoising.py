import parselmouth
from parselmouth.praat import call

def denoise_sound_praat_remove_noise(
    sound: parselmouth.Sound,
    noise_start_s: float,
    noise_end_s: float,
    window_length_s: float = 0.025,
    filter_low_hz: float = 0.0,
    filter_high_hz: float = 20000.0,
    smoothing_hz: float = 40.0,
    method: str = "spectral-subtraction",
) -> parselmouth.Sound:
    """
    Denoise a Parselmouth Sound via Praat's 'Sound: Remove noise...'.

    Notes:
    - noise_start_s, noise_end_s should bracket a region containing ONLY noise.
    - If noise_end_s < noise_start_s, Praat can auto-pick a noise fragment near minimum intensity.  [oai_citation:1‡fon.hum.uva.nl](https://www.fon.hum.uva.nl/praat/manual/Sound__Remove_noise___.html)
    - method must be a string; Praat expects the lowercase label "spectral subtraction".
    - Smoothing hz is applied to the NOISE SPECTRUM ESTIMATE, NOT THE ENTIRE SPEECH SIGNAL.
    """
    
    denoised = call(
        sound,
        "Remove noise...",
        noise_start_s,
        noise_end_s,
        window_length_s,
        filter_low_hz,
        filter_high_hz,
        smoothing_hz,
        method,
    )
    return denoised


snd = parselmouth.Sound("raw_audio/testsoundmono.mp3")

# Pick a noise-only segment (e.g., first 250 ms).
snd_denoised = denoise_sound_praat_remove_noise(
    snd,
    noise_start_s=0.00,
    noise_end_s=0.25,
    window_length_s=0.025,
    filter_low_hz=0.0,
    filter_high_hz=20000.0,
    smoothing_hz=40.0
)

snd_denoised.save("processed_audio/denoised.wav", "WAV")