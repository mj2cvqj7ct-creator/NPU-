from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class AudioFrame:
    """Normalized float PCM frame with non-interleaved metadata-free samples."""

    samples: list[list[float]]
    sample_rate: int

    @property
    def channels(self) -> int:
        return len(self.samples[0]) if self.samples else 0

    @property
    def frame_count(self) -> int:
        return len(self.samples)

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return self.frame_count / self.sample_rate

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not self.samples:
            return
        channels = len(self.samples[0])
        if channels == 0:
            raise ValueError("audio frames must contain at least one channel")
        for row in self.samples:
            if len(row) != channels:
                raise ValueError("all samples must have the same channel count")

    def with_samples(self, samples: list[list[float]]) -> "AudioFrame":
        return AudioFrame(samples=samples, sample_rate=self.sample_rate)

    def scale(self, gain: float) -> "AudioFrame":
        return self.with_samples([[sample * gain for sample in row] for row in self.samples])

    def rms(self) -> float:
        values = [sample for row in self.samples for sample in row]
        if not values:
            return 0.0
        return math.sqrt(sum(sample * sample for sample in values) / len(values))

    def peak(self) -> float:
        values = [abs(sample) for row in self.samples for sample in row]
        return max(values, default=0.0)
