"""Minimal WAV helpers for offline pipeline testing."""

from __future__ import annotations

import wave
from pathlib import Path

from .audio import AudioBuffer


def read_wav(path: str | Path) -> AudioBuffer:
    """Read 16-bit PCM or 32-bit float WAV data into an AudioBuffer."""

    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())

    if sample_width == 2:
        import array

        raw = array.array("h")
        raw.frombytes(frames)
        if raw.itemsize != 2:
            raw.byteswap()
        scale = 1.0 / 32768.0
        samples = [max(-1.0, min(1.0, value * scale)) for value in raw]
    elif sample_width == 4:
        import struct

        count = len(frames) // 4
        samples = list(struct.unpack("<" + "f" * count, frames))
    else:
        raise ValueError(f"unsupported WAV sample width: {sample_width}")

    return AudioBuffer.from_interleaved(samples, sample_rate=sample_rate, channels=channels)


def write_wav(path: str | Path, buffer: AudioBuffer) -> None:
    """Write an AudioBuffer as clipped 16-bit PCM WAV."""

    import array

    clipped = buffer.clipped(-1.0, 1.0)
    pcm = array.array(
        "h",
        int(max(-1.0, min(0.999969482421875, sample)) * 32768.0)
        for sample in clipped.iter_interleaved()
    )

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(buffer.channels)
        wav.setsampwidth(2)
        wav.setframerate(buffer.sample_rate)
        wav.writeframes(pcm.tobytes())
