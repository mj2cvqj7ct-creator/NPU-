"""Small PCM audio container used by the enhancement pipeline.

The production capture layer will eventually feed 48 kHz float32 stereo frames
from WASAPI. Keeping this module dependency-free makes the DSP core easy to
test on non-Windows CI while preserving the same sample layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Callable, Iterable, Sequence


SampleFrame = tuple[float, ...]


@dataclass(frozen=True)
class AudioBuffer:
    """Normalized floating-point PCM frames.

    Samples are stored as ``(frame, channel)`` tuples in the conventional
    ``[-1.0, 1.0]`` range, though intermediate DSP stages may briefly exceed it
    before the limiter is applied.
    """

    sample_rate: int
    channels: int
    frames: tuple[SampleFrame, ...]

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        for frame in self.frames:
            if len(frame) != self.channels:
                raise ValueError("all frames must match channel count")

    @classmethod
    def from_interleaved(
        cls, samples: Sequence[float], sample_rate: int, channels: int
    ) -> "AudioBuffer":
        if channels <= 0:
            raise ValueError("channels must be positive")
        if len(samples) % channels != 0:
            raise ValueError("interleaved sample count is not divisible by channels")

        frames = tuple(
            tuple(float(samples[index + channel]) for channel in range(channels))
            for index in range(0, len(samples), channels)
        )
        return cls(sample_rate=sample_rate, channels=channels, frames=frames)

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def channel_count(self) -> int:
        return self.channels

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.sample_rate

    def copy(self) -> "AudioBuffer":
        return AudioBuffer(self.sample_rate, self.channels, self.frames)

    def as_float32(self) -> "AudioBuffer":
        """Return a normalized float buffer placeholder for capture-host parity."""

        return self.copy()

    def to_interleaved(self) -> tuple[float, ...]:
        return tuple(sample for frame in self.frames for sample in frame)

    def iter_interleaved(self) -> Iterable[float]:
        for frame in self.frames:
            yield from frame

    def mono(self) -> tuple[float, ...]:
        if not self.frames:
            return ()
        return tuple(sum(frame) / self.channels for frame in self.frames)

    def map_samples(self, transform: Callable[[float], float]) -> "AudioBuffer":
        return AudioBuffer(
            sample_rate=self.sample_rate,
            channels=self.channels,
            frames=tuple(tuple(transform(sample) for sample in frame) for frame in self.frames),
        )

    def with_frames(self, frames: Iterable[Sequence[float]]) -> "AudioBuffer":
        return AudioBuffer(
            sample_rate=self.sample_rate,
            channels=self.channels,
            frames=tuple(tuple(float(sample) for sample in frame) for frame in frames),
        )

    def ensure_stereo(self) -> "AudioBuffer":
        if self.channels == 2:
            return self.copy()
        if self.channels == 1:
            return AudioBuffer(
                sample_rate=self.sample_rate,
                channels=2,
                frames=tuple((frame[0], frame[0]) for frame in self.frames),
            )
        return AudioBuffer(
            sample_rate=self.sample_rate,
            channels=2,
            frames=tuple((frame[0], frame[1]) for frame in self.frames),
        )

    def clipped(self, minimum: float, maximum: float) -> "AudioBuffer":
        return self.map_samples(lambda sample: max(minimum, min(maximum, sample)))

    def peak(self) -> float:
        return max((abs(sample) for sample in self.iter_interleaved()), default=0.0)

    def rms(self) -> float:
        samples = self.to_interleaved()
        if not samples:
            return 0.0
        return sqrt(sum(sample * sample for sample in samples) / len(samples))

    def channel_rms(self) -> tuple[float, ...]:
        if not self.frames:
            return tuple(0.0 for _ in range(self.channels))

        sums = [0.0] * self.channels
        for frame in self.frames:
            for channel, sample in enumerate(frame):
                sums[channel] += sample * sample
        return tuple(sqrt(total / self.frame_count) for total in sums)
