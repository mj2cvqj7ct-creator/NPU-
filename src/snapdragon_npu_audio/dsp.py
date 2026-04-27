"""Small, deterministic DSP blocks for real-time music enhancement."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from .audio import AudioBuffer


EPSILON = 1e-12


@dataclass(frozen=True)
class EnhancementMetrics:
    """Fast frame measurements used by DSP and NPU control prediction."""

    peak_dbfs: float
    integrated_lufs: float
    rms_lufs: float
    crest_factor_db: float
    stereo_correlation: float
    low_band_ratio: float
    high_band_ratio: float


@dataclass(frozen=True)
class LimiterSettings:
    ceiling_db: float = -1.0


@dataclass(frozen=True)
class CompressorSettings:
    threshold_db: float = -13.0
    ratio: float = 1.8
    makeup_gain_db: float = 0.0


def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def linear_to_db(value: float) -> float:
    return 20.0 * math.log10(max(abs(value), EPSILON))


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def rms_dbfs(samples: tuple[tuple[float, ...], ...]) -> float:
    flat = [sample for frame in samples for sample in frame]
    if not flat:
        return -120.0
    return linear_to_db(math.sqrt(sum(sample * sample for sample in flat) / len(flat)))


def analyze(buffer: AudioBuffer) -> EnhancementMetrics:
    """Measure audio in terms suitable for short, real-time frames."""

    peak_db = linear_to_db(buffer.peak()) if buffer.frame_count else -120.0
    rms_db = rms_dbfs(buffer.frames)
    low_ratio, high_ratio = _band_ratios(buffer)
    return EnhancementMetrics(
        peak_dbfs=peak_db,
        integrated_lufs=rms_db,
        rms_lufs=rms_db,
        crest_factor_db=max(0.0, peak_db - rms_db),
        stereo_correlation=_stereo_correlation(buffer),
        low_band_ratio=low_ratio,
        high_band_ratio=high_ratio,
    )


def apply_gain(buffer: AudioBuffer, gain_db: float) -> AudioBuffer:
    gain = db_to_linear(gain_db)
    return buffer.map_samples(lambda sample: sample * gain)


def loudness_normalize(
    buffer: AudioBuffer,
    target_lufs: float = -16.0,
    max_gain_db: float = 9.0,
) -> AudioBuffer:
    metrics = analyze(buffer)
    applied_gain = clamp(target_lufs - metrics.integrated_lufs, -max_gain_db, max_gain_db)
    return apply_gain(buffer, applied_gain)


def true_peak_limit(buffer: AudioBuffer, settings: LimiterSettings | None = None) -> AudioBuffer:
    settings = settings or LimiterSettings()
    ceiling = db_to_linear(settings.ceiling_db)
    peak = buffer.peak()
    if peak <= ceiling or peak <= EPSILON:
        return buffer.copy()

    gain = ceiling / peak
    return buffer.map_samples(lambda sample: clamp(sample * gain, -ceiling, ceiling))


def apply_stereo_width_guard(buffer: AudioBuffer, width: float) -> AudioBuffer:
    width = clamp(width, 0.75, 1.18)
    if buffer.channels != 2:
        return buffer.copy()

    frames: list[tuple[float, float]] = []
    for left, right in buffer.frames:
        mid = (left + right) * 0.5
        side = (left - right) * 0.5 * width
        frames.append((mid + side, mid - side))
    return buffer.with_frames(frames)


def apply_multiband_compression(
    buffer: AudioBuffer,
    settings: CompressorSettings | None = None,
) -> AudioBuffer:
    settings = settings or CompressorSettings()
    threshold = db_to_linear(settings.threshold_db)

    def compress(sample: float) -> float:
        magnitude = abs(sample)
        if magnitude <= threshold:
            return sample
        excess = magnitude - threshold
        compressed = threshold + excess / max(settings.ratio, 1.0)
        return math.copysign(compressed, sample)

    return apply_gain(buffer.map_samples(compress), settings.makeup_gain_db)


class OnePoleFilter:
    """Simple first-order low-pass or high-pass section."""

    def __init__(self, sample_rate: int, cutoff_hz: float, mode: str) -> None:
        if mode not in {"lowpass", "highpass"}:
            raise ValueError(f"unsupported filter mode: {mode}")
        self.mode = mode
        rc = 1.0 / (2.0 * math.pi * cutoff_hz)
        dt = 1.0 / sample_rate
        self.alpha = dt / (rc + dt)
        self.low_state = 0.0

    def process(self, sample: float) -> float:
        self.low_state = self.low_state + self.alpha * (sample - self.low_state)
        if self.mode == "lowpass":
            return self.low_state
        return sample - self.low_state


def _parallel_shelf(buffer: AudioBuffer, cutoff_hz: float, gain_db: float, mode: str) -> AudioBuffer:
    if abs(gain_db) < 0.05:
        return buffer.copy()

    band_gain = db_to_linear(gain_db) - 1.0
    filters = [OnePoleFilter(buffer.sample_rate, cutoff_hz, mode) for _ in range(buffer.channels)]
    frames: list[tuple[float, ...]] = []
    for frame in buffer.frames:
        output_frame = []
        for channel_index, sample in enumerate(frame):
            band = filters[channel_index].process(sample)
            output_frame.append(sample + band * band_gain)
        frames.append(tuple(output_frame))
    return buffer.with_frames(frames)


def apply_dynamic_eq(buffer: AudioBuffer, controls: Mapping[str, float]) -> AudioBuffer:
    warmth_db = clamp(float(controls.get("warmth_db", 0.0)), -2.5, 2.5)
    clarity_db = clamp(float(controls.get("clarity_db", 0.0)), -1.5, 2.8)
    air_db = clamp(float(controls.get("air_db", 0.0)), -1.0, 1.8)
    processed = _parallel_shelf(buffer, cutoff_hz=180.0, gain_db=warmth_db, mode="lowpass")
    processed = _parallel_shelf(
        processed,
        cutoff_hz=2_800.0,
        gain_db=clarity_db,
        mode="highpass",
    )
    return _parallel_shelf(processed, cutoff_hz=8_000.0, gain_db=air_db, mode="highpass")


def _band_ratios(buffer: AudioBuffer) -> tuple[float, float]:
    if buffer.frame_count < 2:
        return 0.0, 0.0

    low_energy = EPSILON
    mid_high_energy = EPSILON
    previous_mono = 0.0
    low_filter = OnePoleFilter(buffer.sample_rate, cutoff_hz=350.0, mode="lowpass")

    for frame in buffer.frames:
        mono = sum(frame) / len(frame)
        low = low_filter.process(mono)
        high = mono - previous_mono
        previous_mono = mono
        low_energy += low * low
        mid_high_energy += high * high

    total = low_energy + mid_high_energy
    return low_energy / total, mid_high_energy / total


def _stereo_correlation(buffer: AudioBuffer) -> float:
    if buffer.channels < 2 or not buffer.frames:
        return 1.0
    left = [frame[0] for frame in buffer.frames]
    right = [frame[1] for frame in buffer.frames]
    numerator = sum(l_sample * r_sample for l_sample, r_sample in zip(left, right))
    left_energy = math.sqrt(sum(sample * sample for sample in left))
    right_energy = math.sqrt(sum(sample * sample for sample in right))
    return clamp(numerator / (left_energy * right_energy + EPSILON), -1.0, 1.0)
