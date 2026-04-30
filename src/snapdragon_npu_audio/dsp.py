from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

from .inference import AudioFeatures


StereoFrame = Sequence[Sequence[float]]


@dataclass(frozen=True)
class AudioFrame:
    """Interleaved stereo PCM frame using 32-bit-float-style samples."""

    samples: tuple[float, ...]
    sample_rate: int = 48_000
    channels: int = 2

    @classmethod
    def from_stereo_pairs(
        cls,
        pairs: StereoFrame,
        sample_rate: int = 48_000,
    ) -> "AudioFrame":
        flattened: list[float] = []
        for index, pair in enumerate(pairs):
            if len(pair) != 2:
                raise ValueError(f"sample {index} must contain exactly two channels")
            left = float(pair[0])
            right = float(pair[1])
            if not math.isfinite(left) or not math.isfinite(right):
                raise ValueError(f"sample {index} contains a non-finite value")
            flattened.extend((left, right))
        return cls(tuple(flattened), sample_rate=sample_rate)

    def stereo_pairs(self) -> list[tuple[float, float]]:
        return [
            (self.samples[index], self.samples[index + 1])
            for index in range(0, len(self.samples), 2)
        ]

    def __post_init__(self) -> None:
        if self.channels != 2:
            raise ValueError("only stereo frames are supported")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if len(self.samples) % self.channels != 0:
            raise ValueError("samples must contain complete stereo pairs")
        if any(not math.isfinite(sample) for sample in self.samples):
            raise ValueError("samples must be finite")


@dataclass
class DspState:
    previous_rms: float = 0.0
    gain_smoothing: float = 0.22
    _gain_db: float = 0.0

    def smooth_gain(self, target_gain_db: float) -> float:
        self._gain_db += (target_gain_db - self._gain_db) * self.gain_smoothing
        return self._gain_db


def compute_features(frame: AudioFrame) -> AudioFeatures:
    if not frame.samples:
        return AudioFeatures(
            rms_db=-120.0,
            peak_db=-120.0,
            low_band_energy=1.0,
            mid_band_energy=1.0,
            high_band_energy=1.0,
            transient_score=0.0,
            sample_rate=frame.sample_rate,
        )

    peak = max(abs(sample) for sample in frame.samples)
    rms = math.sqrt(sum(sample * sample for sample in frame.samples) / len(frame.samples))
    low, mid, high = _band_energy(frame.samples)

    return AudioFeatures(
        rms_db=_linear_to_db(rms),
        peak_db=_linear_to_db(peak),
        low_band_energy=low,
        mid_band_energy=mid,
        high_band_energy=high,
        transient_score=_transient_score(frame.samples),
        sample_rate=frame.sample_rate,
    )


def apply_loudness_gain(frame: AudioFrame, gain_db: float) -> AudioFrame:
    gain = _db_to_linear(_clamp(gain_db, -9.0, 9.0))
    return _replace_samples(frame, (sample * gain for sample in frame.samples))


def apply_dynamic_eq(
    frame: AudioFrame,
    low_shelf_db: float,
    presence_db: float,
    sample_rate: int,
) -> AudioFrame:
    low_gain = _db_to_linear(low_shelf_db)
    presence_gain = _db_to_linear(presence_db)
    low_alpha = _one_pole_alpha(180.0, sample_rate)
    presence_alpha = _one_pole_alpha(2_800.0, sample_rate)
    low_state = [0.0, 0.0]
    presence_state = [0.0, 0.0]
    output: list[float] = []

    for index, sample in enumerate(frame.samples):
        channel = index % 2
        low_state[channel] += low_alpha * (sample - low_state[channel])
        presence_state[channel] += presence_alpha * (sample - presence_state[channel])
        high = sample - presence_state[channel]
        mid = sample - low_state[channel] - high
        output.append((low_state[channel] * low_gain) + (mid * presence_gain) + high)

    return _replace_samples(frame, output)


def apply_stereo_width(frame: AudioFrame, width: float) -> AudioFrame:
    width = _clamp(width, 0.75, 1.18)
    output: list[float] = []
    for left, right in frame.stereo_pairs():
        mid = (left + right) * 0.5
        side = (left - right) * 0.5 * width
        output.extend((mid + side, mid - side))
    return _replace_samples(frame, output)


def apply_limiter(frame: AudioFrame, ceiling: float) -> AudioFrame:
    ceiling = _clamp(ceiling, 0.1, 0.999)
    return _replace_samples(frame, (_soft_limiter(sample, ceiling) for sample in frame.samples))


def update_rms_envelope(previous_rms: float, current_rms_db: float, smoothing: float = 0.1) -> float:
    if previous_rms == 0.0:
        return current_rms_db
    return previous_rms + (current_rms_db - previous_rms) * smoothing


def _replace_samples(frame: AudioFrame, samples: Iterable[float]) -> AudioFrame:
    return AudioFrame(tuple(float(sample) for sample in samples), frame.sample_rate, frame.channels)


def _band_energy(samples: Sequence[float]) -> tuple[float, float, float]:
    low_state = [0.0, 0.0]
    presence_state = [0.0, 0.0]
    low = mid = high = 0.0
    low_alpha = 0.015
    presence_alpha = 0.18

    for index, sample in enumerate(samples):
        channel = index % 2
        low_state[channel] += low_alpha * (sample - low_state[channel])
        presence_state[channel] += presence_alpha * (sample - presence_state[channel])
        high_component = sample - presence_state[channel]
        mid_component = sample - low_state[channel] - high_component
        low += abs(low_state[channel])
        mid += abs(mid_component)
        high += abs(high_component)

    average = (low + mid + high) / 3.0
    if average <= 1e-12:
        return (1.0, 1.0, 1.0)
    return (low / average, mid / average, high / average)


def _transient_score(samples: Sequence[float]) -> float:
    if len(samples) < 4:
        return 0.0
    total = 0.0
    jumps = 0.0
    for index in range(2, len(samples)):
        total += abs(samples[index])
        jumps += abs(samples[index] - samples[index - 2])
    if total <= 1e-12:
        return 0.0
    return _clamp(jumps / (total * 4.0), 0.0, 1.0)


def _linear_to_db(value: float) -> float:
    if value <= 1e-12:
        return -120.0
    return 20.0 * math.log10(value)


def _db_to_linear(db: float) -> float:
    return math.pow(10.0, db / 20.0)


def _one_pole_alpha(cutoff_hz: float, sample_rate_hz: int) -> float:
    return 1.0 - math.exp(-2.0 * math.pi * cutoff_hz / sample_rate_hz)


def _apply_stereo_width(left: float, right: float, width: float) -> tuple[float, float]:
    mid = (left + right) * 0.5
    side = (left - right) * 0.5 * width
    return mid + side, mid - side


def _soft_limiter(sample: float, ceiling: float) -> float:
    ceiling = max(0.1, min(ceiling, 1.0))
    if abs(sample) <= ceiling:
        return sample
    sign = 1.0 if sample >= 0.0 else -1.0
    return sign * ceiling


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))

