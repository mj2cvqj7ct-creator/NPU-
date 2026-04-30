from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable


Sample = float
StereoSample = tuple[Sample, Sample]


@dataclass(frozen=True)
class AudioFrame:
    """A short interleaved stereo block normalized to -1.0..1.0 floats."""

    sample_rate: int
    samples: tuple[StereoSample, ...]

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / self.sample_rate

    @property
    def peak(self) -> float:
        return max((max(abs(left), abs(right)) for left, right in self.samples), default=0.0)

    @property
    def rms(self) -> float:
        if not self.samples:
            return 0.0
        energy = sum(left * left + right * right for left, right in self.samples)
        return (energy / (len(self.samples) * 2)) ** 0.5

    def map_samples(self, gain_left: float, gain_right: float) -> "AudioFrame":
        return AudioFrame(
            sample_rate=self.sample_rate,
            samples=tuple((left * gain_left, right * gain_right) for left, right in self.samples),
        )

    def apply_gain_db(self, gain_db: float) -> "AudioFrame":
        gain = 10.0 ** (gain_db / 20.0)
        return self.with_samples((left * gain, right * gain) for left, right in self.samples)

    def with_samples(self, samples: Iterable[StereoSample]) -> "AudioFrame":
        return AudioFrame(
            sample_rate=self.sample_rate,
            samples=tuple((clamp_sample(left), clamp_sample(right)) for left, right in samples),
        )

    def mono(self) -> list[float]:
        return [(left + right) * 0.5 for left, right in self.samples]

    @classmethod
    def from_mono(cls, sample_rate: int, samples: Iterable[Sample]) -> "AudioFrame":
        return cls(sample_rate=sample_rate, samples=tuple((sample, sample) for sample in samples))


def clamp_sample(sample: float, limit: float = 1.0) -> float:
    return max(-limit, min(limit, sample))
