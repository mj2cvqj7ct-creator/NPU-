from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from .audio_frame import AudioFrame


def read_wav(path: str | Path) -> AudioFrame:
    """Read a PCM WAV file into normalized float32 samples."""

    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())

    if sample_width == 1:
        raw = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        samples = (raw - 128.0) / 128.0
    elif sample_width == 2:
        raw = np.frombuffer(frames, dtype="<i2").astype(np.float32)
        samples = raw / 32768.0
    elif sample_width == 3:
        bytes_ = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        sign = (bytes_[:, 2] & 0x80) != 0
        packed = (
            bytes_[:, 0].astype(np.int32)
            | (bytes_[:, 1].astype(np.int32) << 8)
            | (bytes_[:, 2].astype(np.int32) << 16)
        )
        packed[sign] |= ~0xFFFFFF
        samples = packed.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        raw = np.frombuffer(frames, dtype="<i4").astype(np.float32)
        samples = raw / 2147483648.0
    else:
        raise ValueError(f"unsupported WAV sample width: {sample_width}")

    return AudioFrame(samples=samples.reshape(-1, channels), sample_rate=sample_rate)


def write_wav(path: str | Path, frame: AudioFrame, sample_width: int = 2) -> None:
    """Write normalized float32 samples as PCM WAV."""

    if sample_width not in {2, 3, 4}:
        raise ValueError("sample_width must be 2, 3, or 4 bytes")

    clipped = np.clip(frame.samples, -1.0, 1.0)
    if sample_width == 2:
        data = (clipped * 32767.0).astype("<i2").tobytes()
    elif sample_width == 3:
        ints = (clipped.reshape(-1) * 8388607.0).astype(np.int32)
        bytes_ = np.empty((ints.size, 3), dtype=np.uint8)
        bytes_[:, 0] = ints & 0xFF
        bytes_[:, 1] = (ints >> 8) & 0xFF
        bytes_[:, 2] = (ints >> 16) & 0xFF
        data = bytes_.tobytes()
    else:
        data = (clipped * 2147483647.0).astype("<i4").tobytes()

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(frame.channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(frame.sample_rate)
        wav.writeframes(data)
