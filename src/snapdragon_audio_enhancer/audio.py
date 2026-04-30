from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
import wave


FloatFrame = tuple[float, float]


@dataclass(frozen=True)
class AudioBuffer:
    """Interleaved-free stereo PCM buffer using normalized float samples."""

    sample_rate: int
    frames: tuple[FloatFrame, ...]

    @property
    def channels(self) -> int:
        return 2

    @property
    def channel_count(self) -> int:
        return 2

    @property
    def samples(self) -> tuple[FloatFrame, ...]:
        return self.frames

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return len(self.frames) / self.sample_rate

    @property
    def peak(self) -> float:
        return max((max(abs(left), abs(right)) for left, right in self.frames), default=0.0)

    @property
    def rms(self) -> float:
        if not self.frames:
            return 0.0
        total = sum(left * left + right * right for left, right in self.frames)
        return math.sqrt(total / (len(self.frames) * 2))

    def with_frames(self, frames: list[FloatFrame] | tuple[FloatFrame, ...]) -> "AudioBuffer":
        return AudioBuffer(sample_rate=self.sample_rate, frames=tuple(frames))

    def mono_mixdown(self) -> tuple[float, ...]:
        return tuple((left + right) * 0.5 for left, right in self.frames)


def _pcm_to_float(sample: int, width: int) -> float:
    if width == 1:
        return (sample - 128) / 128.0
    max_positive = float(1 << (width * 8 - 1))
    return max(-1.0, sample / max_positive)


def _float_to_pcm(sample: float, width: int) -> int:
    clipped = max(-1.0, min(1.0, sample))
    if width == 1:
        return int(round((clipped * 127.0) + 128))
    max_positive = (1 << (width * 8 - 1)) - 1
    min_negative = -(1 << (width * 8 - 1))
    return max(min_negative, min(max_positive, int(round(clipped * max_positive))))


def read_wav(path: str | Path) -> AudioBuffer:
    """Read a mono or stereo PCM WAV file into the internal stereo float format."""

    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frame_count = wav.getnframes()
        if channels not in (1, 2):
            raise ValueError(f"expected mono or stereo WAV, got {channels} channels")
        if sample_width not in (1, 2, 3, 4):
            raise ValueError(f"unsupported PCM sample width: {sample_width} bytes")

        raw = wav.readframes(frame_count)

    frames: list[FloatFrame] = []
    stride = channels * sample_width
    for offset in range(0, len(raw), stride):
        samples: list[float] = []
        for channel in range(channels):
            start = offset + channel * sample_width
            chunk = raw[start : start + sample_width]
            if sample_width == 1:
                integer = chunk[0]
            else:
                integer = int.from_bytes(chunk, byteorder="little", signed=True)
            samples.append(_pcm_to_float(integer, sample_width))
        if channels == 1:
            frames.append((samples[0], samples[0]))
        else:
            frames.append((samples[0], samples[1]))

    return AudioBuffer(sample_rate=sample_rate, frames=tuple(frames))


def write_wav(path: str | Path, buffer: AudioBuffer, sample_width: int = 2) -> None:
    """Write a stereo PCM WAV file from the internal float format."""

    if sample_width not in (1, 2, 3, 4):
        raise ValueError(f"unsupported PCM sample width: {sample_width} bytes")

    raw = bytearray()
    for left, right in buffer.frames:
        for sample in (left, right):
            integer = _float_to_pcm(sample, sample_width)
            signed = sample_width != 1
            raw.extend(integer.to_bytes(sample_width, byteorder="little", signed=signed))

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(buffer.channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(buffer.sample_rate)
        wav.writeframes(bytes(raw))
