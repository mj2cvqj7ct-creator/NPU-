"""Audio frame primitives used by the enhancer pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence


Sample = tuple[float, float]


@dataclass(frozen=True)
class AudioFormat:
    """Internal processing format for music app output."""

    sample_rate: int = 48_000
    channels: int = 2

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels != 2:
            raise ValueError("this prototype currently supports stereo only")


@dataclass(frozen=True)
class AudioFrame:
    """Interleaved stereo audio represented as normalized float samples."""

    samples: tuple[Sample, ...]
    fmt: AudioFormat = AudioFormat()

    def __post_init__(self) -> None:
        self.fmt.validate()

    @classmethod
    def from_iterable(
        cls, samples: Iterable[Sequence[float]], fmt: AudioFormat | None = None
    ) -> "AudioFrame":
        normalized: list[Sample] = []
        for sample in samples:
            if len(sample) != 2:
                raise ValueError("each sample must contain left and right channels")
            left = _clamp_float(float(sample[0]))
            right = _clamp_float(float(sample[1]))
            normalized.append((left, right))
        return cls(tuple(normalized), fmt or AudioFormat())

    @property
    def duration_seconds(self) -> float:
        return len(self.samples) / self.fmt.sample_rate

    def peak(self) -> float:
        return max((max(abs(left), abs(right)) for left, right in self.samples), default=0.0)

    def iter_interleaved(self) -> Iterator[float]:
        for left, right in self.samples:
            yield left
            yield right


def _clamp_float(value: float) -> float:
    if value > 1.0:
        return 1.0
    if value < -1.0:
        return -1.0
    return value
