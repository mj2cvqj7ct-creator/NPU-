"""Small WAV helpers for offline validation of the realtime processing chain."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def read_wav(path: str | Path) -> tuple[int, np.ndarray]:
    """Read a PCM WAV file and return float32 samples shaped as frames x channels."""

    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        payload = wav.readframes(frames)

    if sample_width == 2:
        raw = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        raw = np.frombuffer(payload, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported PCM sample width: {sample_width}")

    return sample_rate, raw.reshape(-1, channels)


def write_wav(path: str | Path, sample_rate: int, samples: np.ndarray) -> None:
    """Write float samples as 24-bit-safe 32-bit PCM WAV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, np.nextafter(1.0, 0.0))
    pcm = (clipped * 2147483647.0).astype("<i4")

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1 if samples.ndim == 1 else samples.shape[1])
        wav.setsampwidth(4)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
