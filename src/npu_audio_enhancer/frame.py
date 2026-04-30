from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Sequence


Sample = float
StereoSample = tuple[Sample, Sample]


@dataclass(frozen=True)
class AudioFrame:
    """Small 48 kHz stereo frame used by the real-time enhancement chain."""

    samples: tuple[StereoSample, ...]
    sample_rate: int = 48_000

    def __post_init__(self) -> None:
        if self.sample_rate != 48_000:
            raise ValueError("AudioFrame expects the internal 48 kHz sample rate")

        normalised: list[StereoSample] = []
        for sample in self.samples:
            if len(sample) != 2:
                raise ValueError("AudioFrame expects stereo samples")
            left = float(sample[0])
            right = float(sample[1])
            if not isfinite(left) or not isfinite(right):
                raise ValueError("samples must be finite floats")
            normalised.append((left, right))

        object.__setattr__(self, "samples", tuple(normalised))

    @classmethod
    def from_interleaved(
        cls, interleaved: Sequence[Sample], sample_rate: int = 48_000
    ) -> "AudioFrame":
        if len(interleaved) % 2:
            raise ValueError("interleaved stereo buffers must have an even length")

        samples = tuple(
            (float(interleaved[index]), float(interleaved[index + 1]))
            for index in range(0, len(interleaved), 2)
        )
        return cls(samples=samples, sample_rate=sample_rate)

    @classmethod
    def silence(cls, frame_size: int, sample_rate: int = 48_000) -> "AudioFrame":
        if frame_size < 0:
            raise ValueError("frame_size must be non-negative")
        return cls(samples=tuple((0.0, 0.0) for _ in range(frame_size)), sample_rate=sample_rate)

    def to_interleaved(self) -> tuple[Sample, ...]:
        return tuple(value for sample in self.samples for value in sample)

    def map_samples(self, gain: Sample) -> "AudioFrame":
        return AudioFrame(
            samples=tuple((left * gain, right * gain) for left, right in self.samples),
            sample_rate=self.sample_rate,
        )

    def with_samples(self, samples: Iterable[StereoSample]) -> "AudioFrame":
        return AudioFrame(samples=tuple(samples), sample_rate=self.sample_rate)

    @property
    def frame_size(self) -> int:
        return len(self.samples)
