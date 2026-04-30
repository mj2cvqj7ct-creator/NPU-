from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read PCM WAV as float32 with shape (frames, channels)."""

    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        raw = wav.readframes(frames)

    if sample_width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        bytes_ = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        signed = (
            bytes_[:, 0].astype(np.int32)
            | (bytes_[:, 1].astype(np.int32) << 8)
            | (bytes_[:, 2].astype(np.int32) << 16)
        )
        signed = np.where(signed & 0x800000, signed | ~0xFFFFFF, signed)
        data = signed.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    return data.reshape(-1, channels), sample_rate


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write float audio as 24-bit PCM WAV."""

    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.ndim != 2:
        raise ValueError("audio must be mono or channel-interleaved")

    clipped = np.clip(samples, -1.0, 1.0 - (1.0 / 8388608.0))
    pcm = np.round(clipped * 8388608.0).astype(np.int32)
    payload = bytearray()
    for value in pcm.reshape(-1):
        value &= 0xFFFFFF
        payload.extend((value & 0xFF, (value >> 8) & 0xFF, (value >> 16) & 0xFF))

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(samples.shape[1])
        wav.setsampwidth(3)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(payload))
