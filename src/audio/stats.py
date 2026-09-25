from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio(file_path: Path):
    data, sample_rate = sf.read(str(file_path))
    if data.ndim == 2:
        data = data.mean(axis=1)
    return data, sample_rate


def peak_amplitude(samples: np.ndarray):
    return float(np.max(np.abs(samples))) if len(samples) else 0.0


def clipping_ratio(samples: np.ndarray, threshold: float = 0.99):
    if len(samples) == 0:
        return 0.0
    return float(np.mean(np.abs(samples) >= threshold))

def silence_ratio(samples: np.ndarray, threshold: float = 0.05):
    if len(samples) == 0:
        return 0.0
    return float(np.mean(np.abs(samples) <= threshold))