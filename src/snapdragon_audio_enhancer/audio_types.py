from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Callable


Sample = tuple[float, float]


@dataclass(frozen=True)
class AudioFrame:
    """Stereo PCM frame data normalized to -1.0..1.0 float samples."""

    sample_rate: int
    frames: tuple[Sample, ...]
    channels: int = 2

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels != 2:
            raise ValueError("only stereo frames are supported by this prototype")

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.sample_rate

    def peak(self) -> float:
        return max((max(abs(left), abs(right)) for left, right in self.frames), default=0.0)

    def rms(self) -> float:
        if not self.frames:
            return 0.0
        total = sum(left * left + right * right for left, right in self.frames)
        return sqrt(total / (self.frame_count * self.channels))

    def mid_side_ratio(self) -> float:
        if not self.frames:
            return 0.0
        mid_energy = 0.0
        side_energy = 0.0
        for left, right in self.frames:
            mid = (left + right) * 0.5
            side = (left - right) * 0.5
            mid_energy += mid * mid
            side_energy += side * side
        return sqrt(side_energy / max(mid_energy + side_energy, 1.0e-12))

    def map_samples(self, transform: Callable[[float], float]) -> "AudioFrame":
        return AudioFrame(
            sample_rate=self.sample_rate,
            frames=tuple((transform(left), transform(right)) for left, right in self.frames),
            channels=self.channels,
        )

    def chunks(self, frame_size: int) -> tuple["AudioFrame", ...]:
        if frame_size <= 0:
            raise ValueError("frame_size must be positive")
        return tuple(
            AudioFrame(self.sample_rate, self.frames[index : index + frame_size], self.channels)
            for index in range(0, len(self.frames), frame_size)
        )


AudioBuffer = AudioFrame


def clamp_sample(value: float, ceiling: float = 1.0) -> float:
    ceiling = abs(ceiling)
    return max(-ceiling, min(ceiling, value))
