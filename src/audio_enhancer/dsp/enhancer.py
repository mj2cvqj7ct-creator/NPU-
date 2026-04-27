"""Rule-based low-latency audio enhancement primitives.

The production Windows pipeline will receive 48 kHz float stereo frames from
WASAPI.  These functions keep the same constraints while remaining dependency
free, which lets us validate gain staging and limiter behavior before adding
platform-specific capture and NPU backends.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log10, sqrt

from .frame import AudioFrame


EPSILON = 1.0e-12
REFERENCE_LEVEL_DBFS = -23.0


@dataclass(frozen=True)
class EnhancementConfig:
    """Configuration for conservative service-wide audio enhancement."""

    target_loudness_dbfs: float = -16.0
    max_gain_db: float = 9.0
    true_peak_ceiling_dbfs: float = -1.0
    balance_correction_strength: float = 0.35
    vocal_presence_db: float = 1.5
    low_volume_lift_db: float = 2.0
    low_volume_threshold_dbfs: float = -28.0


@dataclass(frozen=True)
class EnhancementMetrics:
    """Frame-level measurements emitted for logging and personalization."""

    input_peak_dbfs: float
    output_peak_dbfs: float
    input_loudness_dbfs: float
    applied_gain_db: float
    balance_delta_db: float
    limiter_gain_db: float


class AudioEnhancer:
    """Small DSP chain for 10-20 ms stereo PCM frames."""

    def __init__(self, config: EnhancementConfig | None = None) -> None:
        self.config = config or EnhancementConfig()

    def process(self, frame: AudioFrame) -> tuple[AudioFrame, EnhancementMetrics]:
        """Apply loudness, balance, clarity, and limiter processing."""

        samples = frame.copy_samples()
        input_peak = _peak_abs(samples)
        input_loudness = _rms_dbfs(samples)

        loudness_gain_db = self._loudness_gain(input_loudness)
        samples = _apply_gain(samples, loudness_gain_db)

        samples, balance_delta_db = self._correct_balance(samples)
        samples = self._apply_presence_and_low_volume_lift(samples, input_loudness)
        limited_samples, limiter_gain_db = self._limit_true_peak(samples)

        output_peak = _peak_abs(limited_samples)
        metrics = EnhancementMetrics(
            input_peak_dbfs=_linear_to_dbfs(input_peak),
            output_peak_dbfs=_linear_to_dbfs(output_peak),
            input_loudness_dbfs=input_loudness,
            applied_gain_db=loudness_gain_db,
            balance_delta_db=balance_delta_db,
            limiter_gain_db=limiter_gain_db,
        )
        return frame.with_samples(limited_samples), metrics

    def _loudness_gain(self, loudness_dbfs: float) -> float:
        if loudness_dbfs <= -120.0:
            return 0.0

        gain = self.config.target_loudness_dbfs - loudness_dbfs
        return _clamp(gain, -self.config.max_gain_db, self.config.max_gain_db)

    def _correct_balance(self, samples: list[tuple[float, float]]) -> tuple[list[tuple[float, float]], float]:
        left_rms, right_rms = _channel_rms(samples)
        if left_rms <= EPSILON or right_rms <= EPSILON:
            return samples, 0.0

        delta_db = 20.0 * log10(left_rms / right_rms)
        correction_db = _clamp(
            delta_db * self.config.balance_correction_strength,
            -3.0,
            3.0,
        )
        left_gain = _db_to_linear(-correction_db / 2.0)
        right_gain = _db_to_linear(correction_db / 2.0)
        corrected = [(left * left_gain, right * right_gain) for left, right in samples]
        return corrected, correction_db

    def _apply_presence_and_low_volume_lift(
        self,
        samples: list[tuple[float, float]],
        input_loudness_dbfs: float,
    ) -> list[tuple[float, float]]:
        lift_db = self.config.vocal_presence_db
        if input_loudness_dbfs < self.config.low_volume_threshold_dbfs:
            lift_db += self.config.low_volume_lift_db

        # A dependency-free approximation of presence enhancement: a gentle
        # high-passed transient component is mixed back at a bounded gain.
        wet_gain = _db_to_linear(lift_db) - 1.0
        previous_mid = 0.0
        enhanced: list[tuple[float, float]] = []
        for left, right in samples:
            mid = (left + right) * 0.5
            transient = mid - previous_mid
            previous_mid = mid
            enhancement = transient * wet_gain * 0.18
            enhanced.append((left + enhancement, right + enhancement))
        return enhanced

    def _limit_true_peak(self, samples: list[tuple[float, float]]) -> tuple[list[tuple[float, float]], float]:
        peak = _peak_abs(samples)
        ceiling = _db_to_linear(self.config.true_peak_ceiling_dbfs)
        if peak <= ceiling or peak <= EPSILON:
            return samples, 0.0

        limiter_gain = ceiling / peak
        return _apply_linear_gain(samples, limiter_gain), 20.0 * log10(limiter_gain)


def _apply_gain(samples: list[tuple[float, float]], gain_db: float) -> list[tuple[float, float]]:
    return _apply_linear_gain(samples, _db_to_linear(gain_db))


def _apply_linear_gain(samples: list[tuple[float, float]], gain: float) -> list[tuple[float, float]]:
    return [(left * gain, right * gain) for left, right in samples]


def _channel_rms(samples: list[tuple[float, float]]) -> tuple[float, float]:
    if not samples:
        return 0.0, 0.0

    left_power = sum(left * left for left, _ in samples) / len(samples)
    right_power = sum(right * right for _, right in samples) / len(samples)
    return sqrt(left_power), sqrt(right_power)


def _rms_dbfs(samples: list[tuple[float, float]]) -> float:
    if not samples:
        return -120.0

    power = sum((left * left + right * right) * 0.5 for left, right in samples) / len(samples)
    return _linear_to_dbfs(sqrt(power))


def _peak_abs(samples: list[tuple[float, float]]) -> float:
    if not samples:
        return 0.0
    return max(max(abs(left), abs(right)) for left, right in samples)


def _linear_to_dbfs(value: float) -> float:
    if value <= EPSILON:
        return -120.0
    return 20.0 * log10(value)


def _db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
