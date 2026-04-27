"""Small WAV reader/writer for offline validation."""

from __future__ import annotations

import wave
from pathlib import Path

from .audio_types import AudioBuffer


def read_wav(path: str | Path) -> AudioBuffer:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        sample_rate = source.getframerate()
        frames = source.getnframes()
        raw = source.readframes(frames)

    if channels not in (1, 2):
        raise ValueError("only mono and stereo WAV files are supported")
    if sample_width not in (2, 3, 4):
        raise ValueError("only 16-bit, 24-bit, and 32-bit PCM WAV files are supported")

    scale = float(1 << (8 * sample_width - 1))
    decoded_frames: list[tuple[float, ...]] = []
    frame_width = channels * sample_width

    for offset in range(0, len(raw), frame_width):
        decoded_channels: list[float] = []
        for channel in range(channels):
            start = offset + channel * sample_width
            value = int.from_bytes(raw[start : start + sample_width], "little", signed=True)
            decoded_channels.append(_clamp_sample(value / scale))
        decoded_frames.append(tuple(decoded_channels))

    if channels == 1:
        stereo_frames = tuple((frame[0], frame[0]) for frame in decoded_frames)
    else:
        stereo_frames = tuple((frame[0], frame[1]) for frame in decoded_frames)
    return AudioBuffer(sample_rate=sample_rate, frames=stereo_frames)


def write_wav(path: str | Path, buffer: AudioBuffer, sample_width: int = 2) -> None:
    if sample_width not in (2, 3, 4):
        raise ValueError("sample_width must be 2, 3, or 4 bytes")

    frames = bytearray()
    max_int = (1 << (8 * sample_width - 1)) - 1
    min_int = -(1 << (8 * sample_width - 1))

    for left, right in buffer.frames:
        for sample in (left, right):
            sample = _clamp_sample(sample)
            value = int(round(sample * max_int))
            value = max(min_int, min(max_int, value))
            frames.extend(value.to_bytes(sample_width, "little", signed=True))

    with wave.open(str(path), "wb") as target:
        target.setnchannels(buffer.channels)
        target.setsampwidth(sample_width)
        target.setframerate(buffer.sample_rate)
        target.writeframes(bytes(frames))


def _clamp_sample(sample: float) -> float:
    return max(-1.0, min(1.0, sample))
