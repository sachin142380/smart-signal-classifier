import numpy as np
from scipy import signal
from scipy.fft import fft
from scipy.signal.windows import gaussian

t = np.linspace(0, 1, 500)

import numpy as np
from scipy import signal

t = np.linspace(0, 1, 500)

def generate_signal(signal_type, freq=5):

    if signal_type == "sine":
        return np.sin(2 * np.pi * freq * t)

    elif signal_type == "square":
        return signal.square(2 * np.pi * freq * t)

    elif signal_type == "triangle":
        return signal.sawtooth(2 * np.pi * freq * t, width=0.5)

    elif signal_type == "sawtooth":
        return signal.sawtooth(2 * np.pi * freq * t)

    elif signal_type == "chirp":
        return signal.chirp(t, f0=1, f1=20, t1=1)

    elif signal_type == "noise":
        return np.random.normal(0, 1, len(t))

    elif signal_type == "pulse":
        return signal.square(2 * np.pi * freq * t, duty=0.3)

    elif signal_type == "damped":
        return np.exp(-5*t) * np.sin(2 * np.pi * freq * t)

    elif signal_type == "am":
        carrier = np.sin(2 * np.pi * 20 * t)
        mod = np.sin(2 * np.pi * freq * t)
        return carrier * mod

    elif signal_type == "fm":
        return np.sin(2 * np.pi * 10 * t + 5 * np.sin(2 * np.pi * freq * t))

    elif signal_type == "gaussian":
        return gaussian(len(t), std=50)

    elif signal_type == "spike":
        sig = np.zeros(len(t))
        sig[len(t)//2] = 1
        return sig

    elif signal_type == "step":
        return np.heaviside(t - 0.5, 1)

    elif signal_type == "random_walk":
        return np.cumsum(np.random.randn(len(t)))

    elif signal_type == "burst":
        sig = np.zeros(len(t))
        sig[200:300] = np.sin(2 * np.pi * freq * t[200:300])
        return sig

    # fallback
    return np.zeros(len(t))

def extract_features(sig):
    features = [
        np.mean(sig),
        np.std(sig),
        np.max(sig),
        np.min(sig)
    ]
    
    fft_vals = np.abs(fft(sig))
    features += [np.mean(fft_vals), np.std(fft_vals)]
    
    return features