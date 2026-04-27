from __future__ import annotations

from dataclasses import dataclass
from math import log10, sqrt
from typing import Iterable, Sequence

StereoSample = tuple[float, float]


def clamp_sample(value: float) -> float:
    return max(-1.0, min(1.0, value))


def linear_to_db(value: float, floor_db: float = -120.0) -> float:
    if value <= 0.0:
        return floor_db
    return 20.0 * log10(value)


@dataclass(frozen=True)
class AudioFrame:
    """A block of normalized stereo PCM samples."""

    sample_rate: int
    samples: tuple[StereoSample, ...]

    @classmethod
    def from_mono(cls, sample_rate: int, samples: Iterable[float]) -> "AudioFrame":
        stereo = tuple((clamp_sample(value), clamp_sample(value)) for value in samples)
        return cls(sample_rate=sample_rate, samples=stereo)

    @classmethod
    def from_interleaved(cls, sample_rate: int, channels: int, values: Sequence[float]) -> "AudioFrame":
        if channels < 1:
            raise ValueError("channels must be positive")
        if len(values) % channels != 0:
            raise ValueError("interleaved sample count is not divisible by channel count")

        frames: list[StereoSample] = []
        for index in range(0, len(values), channels):
            left = clamp_sample(values[index])
            right = clamp_sample(values[index + 1] if channels > 1 else left)
            frames.append((left, right))
        return cls(sample_rate=sample_rate, samples=tuple(frames))

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return len(self.samples) / self.sample_rate

    @property
    def peak(self) -> float:
        return max((max(abs(left), abs(right)) for left, right in self.samples), default=0.0)

    @property
    def rms(self) -> float:
        if not self.samples:
            return 0.0
        total = sum(left * left + right * right for left, right in self.samples)
        return sqrt(total / (2 * len(self.samples)))

    @property
    def rms_db(self) -> float:
        return linear_to_db(self.rms)

    @property
    def peak_db(self) -> float:
        return linear_to_db(self.peak)

    def interleaved(self) -> tuple[float, ...]:
        output: list[float] = []
        for left, right in self.samples:
            output.extend((left, right))
        return tuple(output)
