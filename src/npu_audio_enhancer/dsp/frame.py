from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


FloatBuffer = list[list[float]]


@dataclass(frozen=True)
class AudioFrame:
    """Interleaved-service audio represented as channel-major float samples."""

    samples: FloatBuffer
    sample_rate: int = 48_000

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not self.samples:
            raise ValueError("samples must include at least one channel")

        width = len(self.samples[0])
        if width == 0:
            raise ValueError("each channel must include at least one sample")
        if any(len(channel) != width for channel in self.samples):
            raise ValueError("all channels must have the same sample count")

        normalized = tuple(tuple(float(sample) for sample in channel) for channel in self.samples)
        object.__setattr__(self, "samples", normalized)

    @property
    def channels(self) -> int:
        return len(self.samples)

    @property
    def sample_count(self) -> int:
        return len(self.samples[0])

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / self.sample_rate

    @classmethod
    def from_interleaved(
        cls,
        interleaved_samples: Iterable[float],
        channels: int = 2,
        sample_rate: int = 48_000,
    ) -> "AudioFrame":
        if channels <= 0:
            raise ValueError("channels must be positive")

        values = [float(sample) for sample in interleaved_samples]
        if not values or len(values) % channels != 0:
            raise ValueError("interleaved sample count must be a non-zero multiple of channels")

        channel_major = [values[index::channels] for index in range(channels)]
        return cls(channel_major, sample_rate=sample_rate)

    def to_interleaved(self) -> list[float]:
        interleaved: list[float] = []
        for sample_index in range(self.sample_count):
            for channel in self.samples:
                interleaved.append(channel[sample_index])
        return interleaved

    def with_samples(self, samples: Iterable[Iterable[float]]) -> "AudioFrame":
        return AudioFrame([list(channel) for channel in samples], sample_rate=self.sample_rate)
