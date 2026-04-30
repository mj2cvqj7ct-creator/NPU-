"""Low-latency PCM enhancement primitives.

The routines in this module intentionally avoid third-party DSP dependencies so
they can run in tests on Linux while mapping cleanly to a future ARM64 Windows
WASAPI/APO implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from .profiles import ServiceProfile


MIN_DB = -120.0
TRUE_PEAK_CEILING = 0.98


@dataclass(frozen=True)
class AudioMetrics:
    """Short-time features used by DSP and NPU policy selection."""

    rms_db: float
    peak: float
    crest_factor_db: float
    stereo_width: float
    zero_crossing_rate: float
    clipping_ratio: float


def enhance_frame(
    frame: Iterable[Iterable[float]],
    sample_rate: int,
    settings: EnhancementSettings,
    *,
    target_lufs: float | None = None,
) -> tuple[list[list[float]], AudioMetrics]:
    """Enhance one interleaved stereo frame and return processed samples.

    The pipeline is conservative: normalize toward the service target, apply a
    lightweight four-band tone curve, rebalance stereo width, and finish with a
    soft true-peak limiter. This gives the NPU policy a safe base layer to tune.
    """

    samples = _coerce_stereo(frame)
    metrics = analyze_frame(samples)
    if not samples:
        return samples, metrics

    target = settings.target_lufs if target_lufs is None else target_lufs
    gain_db = _loudness_gain_db(metrics.rms_db, target)
    gained = _apply_gain(samples, gain_db)
    toned = _apply_tone_curve(gained, settings, sample_rate)
    restored = _restore_transients(toned, settings.transient_restore)
    widened = _apply_stereo_width(restored, settings.stereo_width)
    limited = _soft_limiter(widened, ceiling=TRUE_PEAK_CEILING)
    return limited, analyze_frame(limited)


@dataclass(frozen=True)
class EnhancementSettings:
    """Concrete DSP settings after service profile and NPU policy are combined."""

    target_lufs: float
    warmth: float
    presence: float
    air: float
    stereo_width: float
    transient_restore: float

    @classmethod
    def from_profile(
        cls,
        profile: ServiceProfile,
        tuning: object | None = None,
    ) -> "EnhancementSettings":
        clarity = float(getattr(tuning, "presence_gain_db", 0.0))
        warmth = float(getattr(tuning, "warmth_gain_db", 0.0))
        air = float(getattr(tuning, "air_gain_db", 0.0))
        transient = float(getattr(tuning, "transient_boost", 0.0))
        width = float(getattr(tuning, "stereo_width", profile.stereo_width))
        return cls(
            target_lufs=profile.target_lufs,
            warmth=_db_to_linear_delta(profile.bass_gain_db + warmth),
            presence=_db_to_linear_delta(profile.presence_gain_db + clarity),
            air=_db_to_linear_delta(profile.air_gain_db + air),
            stereo_width=_clamp(width, 0.85, 1.20),
            transient_restore=_clamp(profile.transient_restore + transient, 0.0, 0.25),
        )


def analyze_frame(frame: Iterable[Iterable[float]]) -> AudioMetrics:
    """Compute deterministic short-time audio metrics from stereo samples."""

    samples = _coerce_stereo(frame)
    flattened = [sample for pair in samples for sample in pair]
    if not flattened:
        return AudioMetrics(
            rms_db=MIN_DB,
            peak=0.0,
            crest_factor_db=0.0,
            stereo_width=0.0,
            zero_crossing_rate=0.0,
            clipping_ratio=0.0,
        )

    square_mean = sum(sample * sample for sample in flattened) / len(flattened)
    rms = math.sqrt(square_mean)
    peak = max(abs(sample) for sample in flattened)
    rms_db = _linear_to_db(rms)
    crest_factor_db = _linear_to_db(peak / rms) if rms > 0 and peak > 0 else 0.0
    stereo_width = _stereo_width(samples)
    zero_crossing_rate = _zero_crossing_rate(samples)
    clipping_ratio = sum(1 for sample in flattened if abs(sample) >= 0.999) / len(
        flattened
    )

    return AudioMetrics(
        rms_db=rms_db,
        peak=peak,
        crest_factor_db=crest_factor_db,
        stereo_width=stereo_width,
        zero_crossing_rate=zero_crossing_rate,
        clipping_ratio=clipping_ratio,
    )


def _coerce_stereo(frame: Iterable[Iterable[float]]) -> list[list[float]]:
    samples: list[list[float]] = []
    for index, pair in enumerate(frame):
        values = list(pair)
        if len(values) != 2:
            raise ValueError(f"sample {index} must contain exactly two channels")
        left, right = values
        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError(f"sample {index} contains non-finite audio")
        samples.append([float(left), float(right)])
    return samples


def _loudness_gain_db(rms_db: float, target_db: float) -> float:
    if rms_db <= MIN_DB:
        return 0.0
    # Avoid pumping by limiting per-frame correction.
    return max(-9.0, min(9.0, target_db - rms_db))


def _apply_gain(samples: list[list[float]], gain_db: float) -> list[list[float]]:
    gain = 10 ** (gain_db / 20.0)
    return [[left * gain, right * gain] for left, right in samples]


def _apply_tone_curve(
    samples: list[list[float]], settings: EnhancementSettings, sample_rate: int
) -> list[list[float]]:
    """Apply a cheap four-region tilt derived from profile gains.

    This is not a replacement for production biquad filters, but it gives a
    deterministic and low-risk prototype for service/profile policy tests.
    """

    if not samples:
        return []

    low_alpha = _one_pole_alpha(180.0, sample_rate)
    mid_alpha = _one_pole_alpha(1_800.0, sample_rate)
    high_alpha = _one_pole_alpha(6_000.0, sample_rate)

    states = [[0.0, 0.0] for _ in range(3)]
    low_gain = 1.0 + settings.warmth
    presence_gain = 1.0 + settings.presence
    air_gain = 1.0 + settings.air

    processed: list[list[float]] = []
    for left, right in samples:
        out_pair: list[float] = []
        for channel, sample in enumerate((left, right)):
            states[0][channel] = _lowpass(states[0][channel], sample, low_alpha)
            states[1][channel] = _lowpass(states[1][channel], sample, mid_alpha)
            states[2][channel] = _lowpass(states[2][channel], sample, high_alpha)

            low = states[0][channel]
            low_mid = states[1][channel] - low
            presence = states[2][channel] - states[1][channel]
            air = sample - states[2][channel]
            out_pair.append(
                low * low_gain + low_mid + presence * presence_gain + air * air_gain
            )
        processed.append(out_pair)
    return processed


def _restore_transients(
    samples: list[list[float]], amount: float
) -> list[list[float]]:
    if amount <= 0.0 or not samples:
        return samples

    previous = [samples[0][0], samples[0][1]]
    restored: list[list[float]] = []
    for left, right in samples:
        pair = []
        for channel, sample in enumerate((left, right)):
            edge = sample - previous[channel]
            pair.append(sample + edge * amount)
            previous[channel] = sample
        restored.append(pair)
    return restored


def _apply_stereo_width(samples: list[list[float]], width: float) -> list[list[float]]:
    width = max(0.65, min(1.25, width))
    widened: list[list[float]] = []
    for left, right in samples:
        mid = (left + right) * 0.5
        side = (left - right) * 0.5 * width
        widened.append([mid + side, mid - side])
    return widened


def _soft_limiter(samples: list[list[float]], ceiling: float) -> list[list[float]]:
    limited: list[list[float]] = []
    for left, right in samples:
        limited.append([_limit_sample(left, ceiling), _limit_sample(right, ceiling)])
    return limited


def _limit_sample(sample: float, ceiling: float) -> float:
    if abs(sample) <= ceiling:
        return sample
    return math.copysign(ceiling + (1.0 - ceiling) * math.tanh(abs(sample) - ceiling), sample)


def _stereo_width(samples: list[list[float]]) -> float:
    mid_energy = 0.0
    side_energy = 0.0
    for left, right in samples:
        mid = (left + right) * 0.5
        side = (left - right) * 0.5
        mid_energy += mid * mid
        side_energy += side * side
    if mid_energy <= 1e-12:
        return 0.0
    return math.sqrt(side_energy / mid_energy)


def _zero_crossing_rate(samples: list[list[float]]) -> float:
    mono = [(left + right) * 0.5 for left, right in samples]
    if len(mono) < 2:
        return 0.0
    crossings = 0
    previous = mono[0]
    for sample in mono[1:]:
        if (previous < 0 <= sample) or (previous >= 0 > sample):
            crossings += 1
        previous = sample
    return crossings / (len(mono) - 1)


def _one_pole_alpha(cutoff_hz: float, sample_rate: int) -> float:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    dt = 1.0 / sample_rate
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    return dt / (rc + dt)


def _lowpass(previous: float, sample: float, alpha: float) -> float:
    return previous + alpha * (sample - previous)


def _linear_to_db(value: float) -> float:
    if value <= 1e-12:
        return MIN_DB
    return 20.0 * math.log10(value)


def _db_to_linear_delta(value_db: float) -> float:
    return _clamp((10 ** (value_db / 20.0)) - 1.0, -0.25, 0.35)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
