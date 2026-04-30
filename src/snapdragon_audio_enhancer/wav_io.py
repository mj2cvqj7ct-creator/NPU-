"""WAV helpers for the offline prototype CLI and tests."""

from __future__ import annotations

import struct
import wave
from pathlib import Path
from typing import Iterable

from .dsp import SampleFrame, clamp_sample


def read_wav(path: str | Path) -> tuple[int, list[SampleFrame]]:
    """Read mono or stereo PCM WAV into normalized stereo float frames."""

    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        sample_rate = source.getframerate()
        frame_count = source.getnframes()
        raw = source.readframes(frame_count)

    if channels not in {1, 2}:
        raise ValueError(f"unsupported channel count: {channels}")
    if sample_width not in {2, 4}:
        raise ValueError(f"unsupported sample width: {sample_width}")

    frames: list[SampleFrame] = []
    if sample_width == 2:
        values = struct.unpack(f"<{len(raw) // 2}h", raw)
        scale = 32768.0
    else:
        values = struct.unpack(f"<{len(raw) // 4}i", raw)
        scale = 2147483648.0

    if channels == 1:
        for value in values:
            sample = value / scale
            frames.append((sample, sample))
    else:
        for index in range(0, len(values), 2):
            frames.append((values[index] / scale, values[index + 1] / scale))

    return sample_rate, frames


def write_wav(path: str | Path, sample_rate: int, frames: Iterable[SampleFrame]) -> None:
    """Write normalized stereo float frames as 16-bit PCM WAV."""

    payload = bytearray()
    for left, right in frames:
        left_i16 = int(round(clamp_sample(left, 0.999) * 32767.0))
        right_i16 = int(round(clamp_sample(right, 0.999) * 32767.0))
        payload.extend(struct.pack("<hh", left_i16, right_i16))

    with wave.open(str(path), "wb") as target:
        target.setnchannels(2)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(bytes(payload))
