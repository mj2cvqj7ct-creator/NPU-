from __future__ import annotations

import math
from dataclasses import dataclass

from .audio import AudioBuffer, peak
from .profiles import EnhancementProfile, ServiceProfile, mix_profiles


@dataclass(frozen=True)
class PipelineMetrics:
    input_peak: float
    output_peak: float
    input_rms: float
    output_rms: float
    clipped_samples: int


@dataclass(frozen=True)
class EnhancementResult:
    audio: AudioBuffer
    profile: EnhancementProfile
    metrics: PipelineMetrics


def enhance_audio(
    buffer: AudioBuffer,
    profile: EnhancementProfile,
    services: tuple[ServiceProfile, ...] = (),
) -> EnhancementResult:
    """Run the CPU-verifiable version of the realtime post-processing graph."""
    if not buffer.samples:
        empty_metrics = PipelineMetrics(0.0, 0.0, 0.0, 0.0, 0)
        return EnhancementResult(buffer, profile, empty_metrics)

    effective_profile = mix_profiles(profile, services)
    source_peak = peak(buffer.samples)
    source_rms = _rms(buffer.samples)
    if source_peak == 0.0:
        silent_metrics = PipelineMetrics(0.0, 0.0, source_rms, source_rms, 0)
        return EnhancementResult(buffer, effective_profile, silent_metrics)

    normalized = _normalize(buffer.samples, effective_profile.normalize_peak, source_peak)
    equalized = _dynamic_tone_shape(normalized, buffer.channels, effective_profile)
    compressed = [_compress_sample(sample, effective_profile) for sample in equalized]
    widened = _apply_stereo_image(compressed, buffer.channels, effective_profile)
    limited, clipped_count = _true_peak_limit(widened, effective_profile)

    output = AudioBuffer(
        sample_rate=buffer.sample_rate,
        channels=buffer.channels,
        samples=limited,
    )
    metrics = PipelineMetrics(
        input_peak=source_peak,
        output_peak=peak(limited),
        input_rms=source_rms,
        output_rms=_rms(limited),
        clipped_samples=clipped_count,
    )
    return EnhancementResult(output, effective_profile, metrics)


def _normalize(samples: tuple[float, ...], target_peak: float, source_peak: float) -> list[float]:
    gain = target_peak / source_peak
    return [sample * gain for sample in samples]


def _dynamic_tone_shape(
    samples: list[float],
    channels: int,
    profile: EnhancementProfile,
) -> list[float]:
    if not samples:
        return []

    shaped: list[float] = []
    low_memory = [0.0] * max(channels, 1)
    high_memory = [0.0] * max(channels, 1)
    low_alpha = 0.035
    high_alpha = 0.18
    for index, sample in enumerate(samples):
        channel = index % max(channels, 1)
        low_memory[channel] += low_alpha * (sample - low_memory[channel])
        high_memory[channel] += high_alpha * (sample - high_memory[channel])
        low = low_memory[channel]
        high = sample - high_memory[channel]
        body = sample + (low * _db_to_linear_delta(profile.low_shelf_gain_db))
        airy = body + (high * _db_to_linear_delta(profile.air_gain_db))
        vocal = _vocal_presence_curve(airy, profile.presence_gain_db)
        shaped.append(vocal)
    return shaped


def _vocal_presence_curve(sample: float, amount: float) -> float:
    if amount == 0.0:
        return sample
    # A gentle odd-harmonic emphasis approximates presence lift without FFT latency.
    return sample + (math.sin(sample * math.pi) * 0.035 * amount)


def _compress_sample(sample: float, profile: EnhancementProfile) -> float:
    sign = -1.0 if sample < 0 else 1.0
    magnitude = abs(sample)
    if magnitude <= profile.compressor_threshold:
        return sample * profile.makeup_gain

    excess = magnitude - profile.compressor_threshold
    compressed = profile.compressor_threshold + (excess / profile.compressor_ratio)
    return sign * compressed * profile.makeup_gain


def _apply_stereo_image(
    samples: list[float],
    channels: int,
    profile: EnhancementProfile,
) -> list[float]:
    if channels != 2:
        return samples

    enhanced: list[float] = []
    for index in range(0, len(samples), 2):
        left = samples[index]
        right = samples[index + 1]
        mid = (left + right) * 0.5 * profile.vocal_center_focus
        side = (left - right) * 0.5 * profile.stereo_width
        edge = (left - right) * profile.transient_restore * 0.04
        enhanced.extend([mid + side + edge, mid - side - edge])
    return enhanced


def _true_peak_limit(
    samples: list[float],
    profile: EnhancementProfile,
) -> tuple[list[float], int]:
    limit = profile.normalize_peak
    drive = profile.soft_clip_drive
    clipped_count = 0
    limited: list[float] = []
    for sample in samples:
        soft = math.tanh(sample * drive) / math.tanh(drive)
        if abs(soft) > limit:
            clipped_count += 1
            soft = math.copysign(limit, soft)
        limited.append(soft)

    current_peak = peak(limited)
    if current_peak > limit:
        limited = [sample * (limit / current_peak) for sample in limited]
    return limited, clipped_count


def _rms(samples: tuple[float, ...] | list[float]) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def _db_to_linear_delta(db: float) -> float:
    return (10 ** (db / 20.0)) - 1.0
