from __future__ import annotations

import math
from dataclasses import dataclass

from .profiles import EnhancementProfile


@dataclass(frozen=True)
class AudioBuffer:
    sample_rate: int
    channels: int
    samples: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if len(self.samples) % self.channels != 0:
            raise ValueError("sample count must be divisible by channel count")

    @property
    def frames(self) -> int:
        return len(self.samples) // self.channels


@dataclass(frozen=True)
class AudioMetrics:
    peak: float
    rms: float
    loudness_lufs: float
    clipping_events: int


@dataclass(frozen=True)
class EnhancementResult:
    audio: AudioBuffer
    input_metrics: AudioMetrics
    output_metrics: AudioMetrics
    profile: EnhancementProfile
    backend_name: str


def analyze(buffer: AudioBuffer) -> AudioMetrics:
    if not buffer.samples:
        return AudioMetrics(peak=0.0, rms=0.0, loudness_lufs=-120.0, clipping_events=0)

    peak = max(abs(sample) for sample in buffer.samples)
    mean_square = sum(sample * sample for sample in buffer.samples) / len(buffer.samples)
    rms = math.sqrt(mean_square)
    loudness_lufs = 20.0 * math.log10(max(rms, 1e-6)) - 0.691
    clipping_events = sum(1 for sample in buffer.samples if abs(sample) >= 0.999)
    return AudioMetrics(
        peak=peak,
        rms=rms,
        loudness_lufs=loudness_lufs,
        clipping_events=clipping_events,
    )


def enhance_buffer(
    buffer: AudioBuffer,
    profile: EnhancementProfile,
    neural_gains: tuple[float, float, float, float],
    backend_name: str,
) -> EnhancementResult:
    input_metrics = analyze(buffer)
    if not buffer.samples:
        return EnhancementResult(buffer, input_metrics, input_metrics, profile, backend_name)

    bass_gain, presence_gain, air_gain, width_hint = neural_gains
    normalized = _loudness_normalize(buffer.samples, input_metrics, profile)
    tone = _apply_tone_shape(
        normalized,
        buffer.channels,
        profile,
        bass_gain=bass_gain,
        presence_gain=presence_gain,
        air_gain=air_gain,
    )
    compressed = [_compress_sample(sample, profile) for sample in tone]
    restored = _restore_transients(normalized, compressed, profile)
    widened = _apply_stereo_width(restored, buffer.channels, profile, width_hint)
    limited = _true_peak_limit(widened, profile.true_peak_ceiling)
    output = AudioBuffer(buffer.sample_rate, buffer.channels, tuple(limited))
    output_metrics = analyze(output)
    return EnhancementResult(output, input_metrics, output_metrics, profile, backend_name)


def _loudness_normalize(
    samples: tuple[float, ...],
    metrics: AudioMetrics,
    profile: EnhancementProfile,
) -> list[float]:
    if metrics.rms <= 0:
        return list(samples)

    target_rms = 10.0 ** ((profile.target_lufs + 0.691) / 20.0)
    gain = min(profile.max_gain, target_rms / metrics.rms)
    return [sample * gain for sample in samples]


def _apply_tone_shape(
    samples: list[float],
    channels: int,
    profile: EnhancementProfile,
    bass_gain: float,
    presence_gain: float,
    air_gain: float,
) -> list[float]:
    output = samples[:]
    prev = [0.0] * channels
    bass_mix = profile.bass_lift_db / 24.0 + (bass_gain - 1.0) * 0.25
    presence_mix = profile.presence_lift_db / 18.0 + (presence_gain - 1.0) * 0.35
    air_mix = profile.air_lift_db / 18.0 + (air_gain - 1.0) * 0.25

    for index, sample in enumerate(samples):
        channel = index % channels
        low = prev[channel] * 0.985 + sample * 0.015
        high = sample - prev[channel]
        prev[channel] = low
        shaped = sample + low * bass_mix + high * air_mix
        # A smooth odd harmonic lift improves vocal presence without hard clipping.
        shaped += math.tanh(sample * 2.0) * 0.08 * presence_mix
        output[index] = shaped
    return output


def _compress_sample(sample: float, profile: EnhancementProfile) -> float:
    sign = -1.0 if sample < 0 else 1.0
    magnitude = abs(sample)
    if magnitude <= profile.compressor_threshold:
        return sample * profile.makeup_gain

    excess = magnitude - profile.compressor_threshold
    compressed = profile.compressor_threshold + excess / profile.compressor_ratio
    return sign * compressed * profile.makeup_gain


def _apply_stereo_width(
    samples: list[float],
    channels: int,
    profile: EnhancementProfile,
    width_hint: float,
) -> list[float]:
    if channels != 2:
        return samples

    width = min(profile.max_stereo_width, profile.stereo_width * width_hint)
    focused: list[float] = []
    for index in range(0, len(samples), 2):
        left = samples[index]
        right = samples[index + 1]
        mid = (left + right) * 0.5 * profile.center_focus
        side = (left - right) * 0.5 * width
        focused.extend((mid + side, mid - side))
    return focused


def _restore_transients(
    dry_samples: list[float],
    wet_samples: list[float],
    profile: EnhancementProfile,
) -> list[float]:
    if profile.transient_restore <= 0:
        return wet_samples

    restored: list[float] = []
    previous = 0.0
    for dry, wet in zip(dry_samples, wet_samples):
        edge = dry - previous
        previous = dry
        restored.append(wet + edge * profile.transient_restore)
    return restored


def _true_peak_limit(samples: list[float], ceiling: float) -> list[float]:
    peak = max((abs(sample) for sample in samples), default=0.0)
    if peak <= ceiling:
        return [max(-ceiling, min(ceiling, sample)) for sample in samples]

    gain = ceiling / peak
    return [max(-ceiling, min(ceiling, sample * gain)) for sample in samples]
