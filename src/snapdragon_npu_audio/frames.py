"""Audio frame data structures used by the enhancement pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


Number = int | float


@dataclass(frozen=True)
class AudioFrame:
    """A short block of interleaved PCM samples.

    Samples are normalized 32-bit-float-style values in the range [-1.0, 1.0].
    The project uses small frames, typically 10 ms to 20 ms, so this immutable
    structure is intentionally simple and dependency free.
    """

    sample_rate: int
    channels: int
    samples: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if len(self.samples) % self.channels != 0:
            raise ValueError("sample count must be divisible by channel count")
        for sample in self.samples:
            if not -4.0 <= sample <= 4.0:
                raise ValueError("samples must be normalized PCM-like floats")

    @classmethod
    def from_samples(
        cls, sample_rate: int, channels: int, samples: Iterable[Number]
    ) -> "AudioFrame":
        return cls(sample_rate, channels, tuple(float(sample) for sample in samples))

    @property
    def frame_count(self) -> int:
        return len(self.samples) // self.channels

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.sample_rate

    def with_samples(self, samples: Iterable[Number]) -> "AudioFrame":
        return AudioFrame.from_samples(self.sample_rate, self.channels, samples)

    def copy(self) -> "AudioFrame":
        return AudioFrame(self.sample_rate, self.channels, self.samples)

    def channel_samples(self, channel: int) -> tuple[float, ...]:
        if not 0 <= channel < self.channels:
            raise ValueError("channel out of range")
        return self.samples[channel :: self.channels]
