"""Small WAV helpers for offline validation of the real-time DSP chain."""

from __future__ import annotations

import struct
import wave
from pathlib import Path

from .dsp import StereoFrame


def read_wav(path: str | Path) -> tuple[int, list[StereoFrame]]:
    """Read mono or stereo PCM WAV data into normalized float stereo samples."""

    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        raw = wav.readframes(frames)

    if channels not in {1, 2}:
        raise ValueError(f"expected mono or stereo WAV, got {channels} channels")
    if sample_width not in {2, 3, 4}:
        raise ValueError(f"expected 16, 24, or 32-bit PCM WAV, got {sample_width * 8}-bit")

    step = sample_width * channels
    samples_out: list[StereoFrame] = []
    scale = float(1 << (sample_width * 8 - 1))

    for offset in range(0, len(raw), step):
        samples = [
            _decode_pcm(raw[offset + sample_width * channel : offset + sample_width * (channel + 1)])
            / scale
            for channel in range(channels)
        ]
        samples_out.append((samples[0], samples[-1]))

    return sample_rate, samples_out


def write_wav(path: str | Path, sample_rate: int, samples: list[StereoFrame]) -> None:
    """Write normalized float stereo samples as 24-bit PCM WAV."""

    output = bytearray()
    max_value = (1 << 23) - 1
    min_value = -(1 << 23)

    for left, right in samples:
        for sample in (left, right):
            value = max(min_value, min(max_value, round(sample * max_value)))
            output.extend(int(value).to_bytes(3, byteorder="little", signed=True))

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(3)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(output))


def _decode_pcm(raw: bytes) -> int:
    if len(raw) == 2:
        return struct.unpack("<h", raw)[0]
    if len(raw) == 3:
        sign_extension = b"\xff" if raw[-1] & 0x80 else b"\x00"
        return int.from_bytes(raw + sign_extension, byteorder="little", signed=True)
    if len(raw) == 4:
        return struct.unpack("<i", raw)[0]
    raise ValueError(f"unsupported PCM sample width: {len(raw)}")
