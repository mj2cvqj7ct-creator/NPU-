"""Audio frame primitives for the low-latency enhancement pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


Sample = float
StereoSample = tuple[Sample, Sample]


@dataclass(frozen=True)
class AudioFrame:
    """A short block of normalized float stereo PCM samples."""

    sample_rate_hz: int
    channels: int
    samples: tuple[StereoSample, ...]

    @classmethod
    def from_interleaved(
        cls,
        samples: Sequence[float],
        *,
        sample_rate_hz: int = 48_000,
        channels: int = 2,
    ) -> "AudioFrame":
        if channels != 2:
            raise ValueError("AudioFrame currently supports stereo input only")
        if len(samples) % channels != 0:
            raise ValueError("Interleaved sample count must be divisible by channels")

        pairs: list[StereoSample] = []
        for index in range(0, len(samples), channels):
            pairs.append((float(samples[index]), float(samples[index + 1])))
        return cls(sample_rate_hz=sample_rate_hz, channels=channels, samples=tuple(pairs))

    @classmethod
    def silence(
        cls,
        frame_samples: int,
        *,
        sample_rate_hz: int = 48_000,
        channels: int = 2,
    ) -> "AudioFrame":
        if channels != 2:
            raise ValueError("AudioFrame currently supports stereo input only")
        return cls(
            sample_rate_hz=sample_rate_hz,
            channels=channels,
            samples=tuple((0.0, 0.0) for _ in range(frame_samples)),
        )

    def to_interleaved(self) -> list[float]:
        interleaved: list[float] = []
        for left, right in self.samples:
            interleaved.extend((left, right))
        return interleaved

    def map_samples(self, values: Iterable[StereoSample]) -> "AudioFrame":
        return AudioFrame(
            sample_rate_hz=self.sample_rate_hz,
            channels=self.channels,
            samples=tuple(values),
        )

    def copy_samples(self) -> list[StereoSample]:
        return list(self.samples)

    def with_samples(self, samples: Iterable[StereoSample]) -> "AudioFrame":
        return self.map_samples(samples)

    @property
    def frame_samples(self) -> int:
        return len(self.samples)

    @property
    def duration_ms(self) -> float:
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        return self.frame_samples * 1000.0 / self.sample_rate_hz

    def channel_values(self, channel_index: int) -> List[float]:
        if channel_index == 0:
            return [left for left, _ in self.samples]
        if channel_index == 1:
            return [right for _, right in self.samples]
        raise ValueError("channel_index must be 0 or 1")
