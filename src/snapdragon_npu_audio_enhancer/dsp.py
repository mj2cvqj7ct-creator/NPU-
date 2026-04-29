from __future__ import annotations

import math
from dataclasses import replace
from typing import Iterable

from .models import AudioAnalysis, EnhancementConfig, EnhancementReport, Frame
from .service_profiles import resolve_service_profile


def enhance_frame(
    frame: Iterable[tuple[float, float]],
    *,
    config: EnhancementConfig,
    provider: object | None = None,
) -> tuple[Frame, EnhancementReport]:
    """Enhance one interleaved stereo frame without retaining stream history."""
    samples = [(float(left), float(right)) for left, right in frame]
    if not samples:
        return [], EnhancementReport.empty(config)

    effective_config = _apply_service_profile(config)
    analysis = _analyze(samples)
    working = _remove_dc(samples)
    working = _balance_channels(working)
    working = _dynamic_eq(working, analysis, effective_config)

    if provider is not None and getattr(provider, "available", False):
        working = provider.enhance(working, effective_config, analysis)

    working, limiter_reduction_db = _true_peak_limit(working, effective_config.true_peak_ceiling)

    report = EnhancementReport(
        sample_rate=effective_config.sample_rate,
        service=effective_config.service,
        input_peak=analysis.peak,
        output_peak=max(max(abs(left), abs(right)) for left, right in working),
        input_rms_db=_linear_to_db(analysis.rms),
        output_rms_db=_linear_to_db(_rms(working)),
        limiter_reduction_db=limiter_reduction_db,
        provider=getattr(provider, "name", "rule-based") if provider is not None else "rule-based",
        npu_accelerated=bool(getattr(provider, "npu_accelerated", False)),
    )
    return working, report


def _apply_service_profile(config: EnhancementConfig) -> EnhancementConfig:
    profile = resolve_service_profile(config.service)
    return replace(
        config,
        clarity=config.clarity * profile.clarity,
        warmth=config.warmth * profile.warmth,
        stereo_width=config.stereo_width * profile.stereo_width,
        target_lufs=config.target_lufs + profile.loudness_offset_db,
    )


def _analyze(frame: Frame) -> AudioAnalysis:
    mono = [(left + right) * 0.5 for left, right in frame]
    rms = _rms(frame)
    peak = max(max(abs(left), abs(right)) for left, right in frame)
    low, mid, high = _split_bands(mono)
    return AudioAnalysis(
        rms=max(rms, 1e-9),
        peak=peak,
        crest_factor_db=_linear_to_db(peak / max(rms, 1e-9)),
        low_energy=_mono_rms(low),
        mid_energy=_mono_rms(mid),
        high_energy=_mono_rms(high),
        stereo_correlation=_stereo_correlation(frame),
        zero_crossing_rate=_zero_crossing_rate(mono),
    )


def _remove_dc(frame: Frame) -> Frame:
    left_mean = sum(left for left, _ in frame) / len(frame)
    right_mean = sum(right for _, right in frame) / len(frame)
    return [(left - left_mean, right - right_mean) for left, right in frame]


def _balance_channels(frame: Frame) -> Frame:
    left_rms = math.sqrt(sum(left * left for left, _ in frame) / len(frame))
    right_rms = math.sqrt(sum(right * right for _, right in frame) / len(frame))
    if left_rms <= 1e-9 or right_rms <= 1e-9:
        return frame
    ratio = math.sqrt(right_rms / left_rms)
    return [(left * ratio, right / ratio) for left, right in frame]


def _dynamic_eq(frame: Frame, analysis: AudioAnalysis, config: EnhancementConfig) -> Frame:
    target_linear = _db_to_linear(config.target_lufs)
    loudness_gain = _soft_gain(target_linear / max(analysis.rms, 1e-9), limit_db=9.0)
    brightness_deficit = max(0.0, analysis.mid_energy - analysis.high_energy)
    warmth_deficit = max(0.0, analysis.mid_energy - analysis.low_energy)

    high_gain = 1.0 + min(0.18, brightness_deficit * config.clarity * 0.8)
    low_gain = 1.0 + min(0.12, warmth_deficit * config.warmth * 0.5)

    mono = [(left + right) * 0.5 for left, right in frame]
    low, _, high = _split_bands(mono)
    enhanced: Frame = []
    side_gain = max(0.75, min(1.35, config.stereo_width))

    for index, (left, right) in enumerate(frame):
        mid = (left + right) * 0.5
        side = (left - right) * 0.5 * side_gain
        tone = low[index] * (low_gain - 1.0) + high[index] * (high_gain - 1.0)
        enhanced.append(((mid + tone + side) * loudness_gain, (mid + tone - side) * loudness_gain))
    return enhanced


def _true_peak_limit(frame: Frame, ceiling: float) -> tuple[Frame, float]:
    peak = max(max(abs(left), abs(right)) for left, right in frame)
    if peak <= ceiling or peak <= 1e-9:
        return frame, 0.0
    gain = ceiling / peak
    return [(left * gain, right * gain) for left, right in frame], abs(_linear_to_db(gain))


def _split_bands(mono: list[float]) -> tuple[list[float], list[float], list[float]]:
    low: list[float] = []
    high: list[float] = []
    low_state = 0.0
    high_state = 0.0
    previous = mono[0] if mono else 0.0

    for sample in mono:
        low_state = 0.92 * low_state + 0.08 * sample
        transient = sample - previous
        high_state = 0.55 * high_state + 0.45 * transient
        low.append(low_state)
        high.append(high_state)
        previous = sample

    mid = [sample - low_part - high_part for sample, low_part, high_part in zip(mono, low, high)]
    return low, mid, high


def _rms(frame: Frame) -> float:
    return math.sqrt(sum(left * left + right * right for left, right in frame) / (len(frame) * 2))


def _mono_rms(samples: list[float]) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def _stereo_correlation(frame: Frame) -> float:
    left_energy = sum(left * left for left, _ in frame)
    right_energy = sum(right * right for _, right in frame)
    if left_energy <= 1e-12 or right_energy <= 1e-12:
        return 0.0
    cross = sum(left * right for left, right in frame)
    return max(-1.0, min(1.0, cross / math.sqrt(left_energy * right_energy)))


def _zero_crossing_rate(samples: list[float]) -> float:
    if len(samples) < 2:
        return 0.0
    crossings = 0
    previous = samples[0]
    for sample in samples[1:]:
        if (previous < 0.0 <= sample) or (previous >= 0.0 > sample):
            crossings += 1
        previous = sample
    return crossings / (len(samples) - 1)


def _soft_gain(gain: float, *, limit_db: float) -> float:
    limit = _db_to_linear(limit_db)
    if gain <= limit:
        return gain
    return limit + math.log1p(gain - limit) * 0.25


def _db_to_linear(db_value: float) -> float:
    return 10 ** (db_value / 20.0)


def _linear_to_db(value: float) -> float:
    return 20.0 * math.log10(max(value, 1e-12))
