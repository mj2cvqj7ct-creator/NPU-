from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


StereoFrame = list[tuple[float, float]]


@dataclass(frozen=True)
class AudioMetrics:
    """Short-window measurements used to drive safe enhancement decisions."""

    rms_dbfs: float
    rms: float
    peak_dbfs: float
    peak: float
    crest_factor_db: float
    clipping_ratio: float
    stereo_correlation: float
    channel_imbalance: float
    spectral_density: float


@dataclass(frozen=True)
class AudioFeatures:
    """Compact feature vector suitable for a tiny NPU control model."""

    rms: float
    peak: float
    crest_factor: float
    clipping_ratio: float
    stereo_correlation: float
    channel_imbalance: float
    spectral_density: float


@dataclass(frozen=True)
class EnhancementSettings:
    target_loudness_dbfs: float = -18.0
    max_loudness_gain_db: float = 6.0
    limiter_ceiling_dbfs: float = -1.0
    clarity_gain_db: float = 1.5
    warmth_gain_db: float = 0.8
    stereo_width: float = 1.08


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def linear_to_db(value: float) -> float:
    if value <= 0.0:
        return -120.0
    return 20.0 * math.log10(value)


def measure_frame(frame: Iterable[tuple[float, float]]) -> AudioMetrics:
    samples = list(frame)
    if not samples:
        return AudioMetrics(-120.0, 0.0, -120.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    energy = 0.0
    left_energy = 0.0
    right_energy = 0.0
    high_energy = 0.0
    peak = 0.0
    clipped = 0
    sum_mid = 0.0
    sum_side = 0.0
    sum_mid_sq = 0.0
    sum_side_sq = 0.0

    prev_left, prev_right = samples[0]
    for left, right in samples:
        left_energy += left * left
        right_energy += right * right
        high_energy += (left - prev_left) * (left - prev_left)
        high_energy += (right - prev_right) * (right - prev_right)
        prev_left = left
        prev_right = right

        for sample in (left, right):
            abs_sample = abs(sample)
            energy += sample * sample
            peak = max(peak, abs_sample)
            if abs_sample >= 0.999:
                clipped += 1

        mid = 0.5 * (left + right)
        side = 0.5 * (left - right)
        sum_mid += mid
        sum_side += side
        sum_mid_sq += mid * mid
        sum_side_sq += side * side

    sample_count = len(samples) * 2
    rms = math.sqrt(energy / sample_count)
    rms_dbfs = linear_to_db(rms)
    peak_dbfs = linear_to_db(peak)
    crest_factor_db = peak_dbfs - rms_dbfs if rms > 0.0 else 0.0

    mid_variance = max(sum_mid_sq / len(samples) - (sum_mid / len(samples)) ** 2, 0.0)
    side_variance = max(sum_side_sq / len(samples) - (sum_side / len(samples)) ** 2, 0.0)
    if mid_variance + side_variance == 0.0:
        stereo_correlation = 0.0
    else:
        stereo_correlation = clamp(
            (mid_variance - side_variance) / (mid_variance + side_variance),
            -1.0,
            1.0,
        )

    left_rms = math.sqrt(left_energy / len(samples))
    right_rms = math.sqrt(right_energy / len(samples))
    channel_imbalance = abs(left_rms - right_rms) / max(left_rms, right_rms, 1e-9)
    spectral_density = clamp(math.sqrt(high_energy / sample_count) / max(rms, 1e-9), 0.0, 1.0)

    return AudioMetrics(
        rms_dbfs=rms_dbfs,
        rms=rms,
        peak_dbfs=peak_dbfs,
        peak=peak,
        crest_factor_db=crest_factor_db,
        clipping_ratio=clipped / sample_count,
        stereo_correlation=stereo_correlation,
        channel_imbalance=channel_imbalance,
        spectral_density=spectral_density,
    )


def extract_features(frame: StereoFrame, metrics: AudioMetrics | None = None) -> AudioFeatures:
    """Extract frame-local controls without retaining or exporting audio."""

    measured = metrics or measure_frame(frame)
    crest_ratio = measured.peak / max(measured.rms, 1e-9)
    return AudioFeatures(
        rms=measured.rms,
        peak=measured.peak,
        crest_factor=crest_ratio,
        clipping_ratio=measured.clipping_ratio,
        stereo_correlation=measured.stereo_correlation,
        channel_imbalance=measured.channel_imbalance,
        spectral_density=measured.spectral_density,
    )


def apply_loudness_gain(
    frame: StereoFrame, metrics: AudioMetrics, settings: EnhancementSettings
) -> StereoFrame:
    desired_gain_db = settings.target_loudness_dbfs - metrics.rms_dbfs
    gain_db = clamp(desired_gain_db, -3.0, settings.max_loudness_gain_db)

    peak_after_gain = metrics.peak_dbfs + gain_db
    if peak_after_gain > settings.limiter_ceiling_dbfs:
        gain_db -= peak_after_gain - settings.limiter_ceiling_dbfs

    gain = db_to_linear(gain_db)
    return [(left * gain, right * gain) for left, right in frame]


def apply_tone_shaping(
    frame: StereoFrame,
    settings: EnhancementSettings,
    clarity_amount: float,
    warmth_amount: float,
) -> StereoFrame:
    if not frame:
        return []

    clarity_gain = db_to_linear(settings.clarity_gain_db * clamp(clarity_amount, 0.0, 1.0))
    warmth_gain = db_to_linear(settings.warmth_gain_db * clamp(warmth_amount, 0.0, 1.0))

    shaped: StereoFrame = []
    prev_left = frame[0][0]
    prev_right = frame[0][1]
    low_left = prev_left
    low_right = prev_right

    for left, right in frame:
        low_left = 0.92 * low_left + 0.08 * left
        low_right = 0.92 * low_right + 0.08 * right

        high_left = left - prev_left
        high_right = right - prev_right
        prev_left = left
        prev_right = right

        shaped_left = left + (low_left * (warmth_gain - 1.0)) + (high_left * (clarity_gain - 1.0))
        shaped_right = right + (low_right * (warmth_gain - 1.0)) + (high_right * (clarity_gain - 1.0))
        shaped.append((shaped_left, shaped_right))

    return shaped


def apply_stereo_width(
    frame: StereoFrame, settings: EnhancementSettings, metrics: AudioMetrics
) -> StereoFrame:
    if metrics.stereo_correlation < 0.15:
        width = min(settings.stereo_width, 1.02)
    else:
        width = settings.stereo_width

    widened: StereoFrame = []
    for left, right in frame:
        mid = 0.5 * (left + right)
        side = 0.5 * (left - right) * width
        widened.append((mid + side, mid - side))
    return widened


def apply_transient_restore(frame: StereoFrame, amount: float) -> StereoFrame:
    """Add a small attack emphasis while avoiding codec-artifact exaggeration."""

    bounded_amount = clamp(amount, 0.0, 0.3)
    if not frame or bounded_amount == 0.0:
        return list(frame)

    restored: StereoFrame = []
    prev_left, prev_right = frame[0]
    for left, right in frame:
        attack_left = left - prev_left
        attack_right = right - prev_right
        prev_left = left
        prev_right = right
        restored.append(
            (
                left + attack_left * bounded_amount,
                right + attack_right * bounded_amount,
            )
        )
    return restored


def apply_true_peak_limiter(frame: StereoFrame, ceiling_dbfs: float = -1.0) -> StereoFrame:
    ceiling = db_to_linear(ceiling_dbfs)
    limited: StereoFrame = []
    for left, right in frame:
        limited.append((_soft_clip(left, ceiling), _soft_clip(right, ceiling)))
    return limited


def _soft_clip(sample: float, ceiling: float) -> float:
    if abs(sample) <= ceiling:
        return sample
    sign = 1.0 if sample >= 0.0 else -1.0
    overshoot = abs(sample) - ceiling
    return sign * min(ceiling, ceiling + math.tanh(overshoot) * (1.0 - ceiling) * 0.1)
