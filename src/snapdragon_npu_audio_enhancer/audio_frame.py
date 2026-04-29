from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence


StereoSamples = tuple[list[float], list[float]]


@dataclass(frozen=True)
class AudioFrame:
    """48 kHz-style floating point PCM container used by the prototype pipeline."""

    stereo_samples: StereoSamples
    sample_rate: int = 48_000

    def __post_init__(self) -> None:
        left, right = self.stereo_samples
        if len(left) != len(right):
            raise ValueError("left and right channels must have the same sample count")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        object.__setattr__(self, "stereo_samples", (list(left), list(right)))

    @classmethod
    def from_interleaved(
        cls, samples: Sequence[float], *, sample_rate: int = 48_000, channels: int = 2
    ) -> "AudioFrame":
        if channels != 2:
            raise ValueError("the prototype currently supports stereo PCM only")
        if len(samples) % channels:
            raise ValueError("interleaved sample count is not divisible by channel count")
        left = [float(samples[index]) for index in range(0, len(samples), 2)]
        right = [float(samples[index]) for index in range(1, len(samples), 2)]
        return cls((left, right), sample_rate=sample_rate)

    @property
    def channel_count(self) -> int:
        return 2

    @property
    def channels(self) -> int:
        return self.channel_count

    @property
    def frame_count(self) -> int:
        return len(self.stereo_samples[0])

    @property
    def samples(self) -> list[tuple[float, float]]:
        left, right = self.stereo_samples
        return list(zip(left, right, strict=True))

    @property
    def peak(self) -> float:
        return max((abs(sample) for sample in self.iter_samples()), default=0.0)

    @property
    def peak_dbfs(self) -> float:
        return self.amplitude_to_db(self.peak)

    @property
    def rms(self) -> float:
        values = list(self.iter_samples())
        if not values:
            return 0.0
        return math.sqrt(sum(sample * sample for sample in values) / len(values))

    @property
    def rms_dbfs(self) -> float:
        return self.rms_db()

    def rms_db(self) -> float:
        return self.amplitude_to_db(self.rms)

    def iter_samples(self) -> Iterable[float]:
        left, right = self.stereo_samples
        yield from left
        yield from right

    def iter_mono(self) -> Iterable[float]:
        left, right = self.stereo_samples
        for l_sample, r_sample in zip(left, right, strict=True):
            yield (l_sample + r_sample) * 0.5

    def iter_stereo(self) -> Iterable[tuple[float, float]]:
        left, right = self.stereo_samples
        yield from zip(left, right, strict=True)

    def mono(self) -> list[float]:
        return list(self.iter_mono())

    def interleaved(self) -> list[float]:
        interleaved: list[float] = []
        left, right = self.stereo_samples
        for l_sample, r_sample in zip(left, right, strict=True):
            interleaved.extend((l_sample, r_sample))
        return interleaved

    def with_stereo_samples(self, left: Sequence[float], right: Sequence[float]) -> "AudioFrame":
        return AudioFrame((list(left), list(right)), sample_rate=self.sample_rate)

    def with_samples(self, samples: Sequence[tuple[float, float]]) -> "AudioFrame":
        left = [float(sample[0]) for sample in samples]
        right = [float(sample[1]) for sample in samples]
        return self.with_stereo_samples(left, right)

    @classmethod
    def from_stereo_pairs(
        cls, samples: Sequence[tuple[float, float]], *, sample_rate: int = 48_000
    ) -> "AudioFrame":
        left = [float(sample[0]) for sample in samples]
        right = [float(sample[1]) for sample in samples]
        return cls((left, right), sample_rate=sample_rate)

    def apply_gain(self, gain: float) -> "AudioFrame":
        left, right = self.stereo_samples
        return self.with_stereo_samples(
            [sample * gain for sample in left],
            [sample * gain for sample in right],
        )

    def apply_gain_db(self, gain_db: float) -> "AudioFrame":
        return self.apply_gain(self.db_to_amplitude(gain_db))

    def split(self, frame_size: int) -> list["AudioFrame"]:
        if frame_size <= 0:
            raise ValueError("frame_size must be positive")
        left, right = self.stereo_samples
        return [
            self.with_stereo_samples(left[start : start + frame_size], right[start : start + frame_size])
            for start in range(0, self.frame_count, frame_size)
        ]

    @staticmethod
    def db_to_amplitude(db: float) -> float:
        return 10 ** (db / 20.0)

    @staticmethod
    def amplitude_to_db(amplitude: float) -> float:
        if amplitude <= 1e-12:
            return -120.0
        return 20.0 * math.log10(amplitude)


@dataclass(frozen=True)
class ProcessingMetrics:
    input_rms_db: float
    output_rms_db: float
    input_peak_db: float
    output_peak_db: float
    provider: str
    frame_count: int

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "input_rms_db": self.input_rms_db,
            "output_rms_db": self.output_rms_db,
            "input_peak_db": self.input_peak_db,
            "output_peak_db": self.output_peak_db,
            "provider": self.provider,
            "frame_count": self.frame_count,
        }
