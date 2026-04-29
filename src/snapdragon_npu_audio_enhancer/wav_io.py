from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from .audio_frame import AudioFrame


def read_wav(path: str | Path) -> AudioFrame:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())

    if channels <= 0:
        raise ValueError("WAV file must contain at least one channel")

    if sample_width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        signed = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        signed = np.where(signed & 0x800000, signed | ~0xFFFFFF, signed)
        data = signed.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported PCM sample width: {sample_width} bytes")

    samples = data.reshape(-1, channels)
    if channels == 1:
        samples = np.repeat(samples, 2, axis=1)
    elif channels > 2:
        samples = samples[:, :2]
    return AudioFrame(samples=samples, sample_rate=sample_rate)


def write_wav(path: str | Path, frame: AudioFrame) -> None:
    clipped = np.clip(frame.samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(frame.channels)
        handle.setsampwidth(2)
        handle.setframerate(frame.sample_rate)
        handle.writeframes(pcm.tobytes())
