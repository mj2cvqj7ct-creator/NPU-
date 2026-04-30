"""Shared audio data types for the prototype pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


AudioSamples = list[list[float]]


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def clamp_sample(value: float) -> float:
    return clamp(value, -1.0, 1.0)


@dataclass(frozen=True)
class AudioBuffer:
    """Interleaved-by-frame floating point PCM.

    Each frame contains one sample per channel. The realtime target is
    48 kHz stereo float PCM, but the offline tool accepts mono or stereo WAVs.
    """

    samples: AudioSamples
    sample_rate: int = 48_000
    channels: int = 2

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels not in (1, 2):
            raise ValueError("only mono or stereo audio is supported")
        for frame in self.samples:
            if len(frame) != self.channels:
                raise ValueError("each frame must match the channel count")

    def copy_with(self, samples: AudioSamples) -> "AudioBuffer":
        return AudioBuffer(samples=samples, sample_rate=self.sample_rate, channels=self.channels)


@dataclass(frozen=True)
class AudioFrame:
    peak: float
    rms: float
    transient_density: float
    low_energy: float
    vocal_energy: float


def peak(samples: Sequence[Sequence[float]]) -> float:
    if not samples:
        return 0.0
    return max((abs(sample) for frame in samples for sample in frame), default=0.0)


def rms(samples: Sequence[Sequence[float]]) -> float:
    count = sum(len(frame) for frame in samples)
    if count == 0:
        return 0.0
    total = sum(sample * sample for frame in samples for sample in frame)
    return (total / count) ** 0.5
