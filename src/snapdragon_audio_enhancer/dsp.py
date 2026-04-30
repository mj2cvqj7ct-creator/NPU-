from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from .audio_types import AudioFrame, clamp_sample
from .inference import EnhancementControls
from .profile import EnhancementProfile


@dataclass(frozen=True)
class FrameMetrics:
    rms: float
    peak: float
    low_energy: float
    mid_energy: float
    high_energy: float


class OnePoleFilter:
    def __init__(self, alpha: float) -> None:
        self.alpha = max(0.0, min(1.0, alpha))
        self.previous = 0.0

    def low_pass(self, sample: float) -> float:
        self.previous += self.alpha * (sample - self.previous)
        return self.previous


class DynamicEq:
    def __init__(self, profile: EnhancementProfile, sample_rate: int) -> None:
        low_alpha = min(1.0, 220.0 / sample_rate)
        high_alpha = min(1.0, 4200.0 / sample_rate)
        self.profile = profile
        self.low_filters = [OnePoleFilter(low_alpha), OnePoleFilter(low_alpha)]
        self.high_filters = [OnePoleFilter(high_alpha), OnePoleFilter(high_alpha)]

    def process(self, buffer: AudioFrame, controls: EnhancementControls) -> AudioFrame:
        processed: list[tuple[float, float]] = []
        low_gain = self.profile.bass_gain * (1.0 + 0.18 * controls.bass_tightness)
        presence_gain = self.profile.presence_gain + 0.12 * controls.clarity
        air_gain = self.profile.air_gain + 0.05 * controls.transient_restore
        width = min(1.15, self.profile.stereo_width + 0.08 * controls.stereo_width)

        for left, right in buffer.frames:
            low_left = self.low_filters[0].low_pass(left)
            low_right = self.low_filters[1].low_pass(right)
            smooth_left = self.high_filters[0].low_pass(left)
            smooth_right = self.high_filters[1].low_pass(right)
            high_left = left - smooth_left
            high_right = right - smooth_right

            enhanced_left = left + low_left * low_gain + (left - low_left - high_left) * presence_gain + high_left * air_gain
            enhanced_right = right + low_right * low_gain + (right - low_right - high_right) * presence_gain + high_right * air_gain

            mid = (enhanced_left + enhanced_right) * 0.5
            side = (enhanced_left - enhanced_right) * 0.5 * width
            processed.append((mid + side, mid - side))

        return AudioFrame(sample_rate=buffer.sample_rate, channels=buffer.channels, frames=tuple(processed))


class LoudnessNormalizer:
    def __init__(self, target_rms: float = 0.18, max_gain: float = 2.4) -> None:
        self.target_rms = target_rms
        self.max_gain = max_gain

    def process(self, buffer: AudioFrame, metrics: FrameMetrics) -> AudioFrame:
        if metrics.rms <= 1e-9:
            return buffer
        gain = min(self.max_gain, self.target_rms / metrics.rms)
        return buffer.map_samples(lambda sample: sample * gain)


class TruePeakLimiter:
    def __init__(self, ceiling: float = 0.98) -> None:
        self.ceiling = ceiling

    def process(self, buffer: AudioFrame) -> AudioFrame:
        peak = max((abs(sample) for frame in buffer.frames for sample in frame), default=0.0)
        if peak <= self.ceiling:
            return buffer
        gain = self.ceiling / peak
        return buffer.map_samples(lambda sample: clamp_sample(sample * gain, self.ceiling))


def analyze_frame(buffer: AudioFrame) -> FrameMetrics:
    if not buffer.frames:
        return FrameMetrics(rms=0.0, peak=0.0, low_energy=0.0, mid_energy=0.0, high_energy=0.0)

    sample_count = len(buffer.frames) * buffer.channels
    total_square = sum(sample * sample for frame in buffer.frames for sample in frame)
    peak = max(abs(sample) for frame in buffer.frames for sample in frame)

    low_total = 0.0
    mid_total = 0.0
    high_total = 0.0
    previous_mid = 0.0
    for left, right in buffer.frames:
        mid = (left + right) * 0.5
        delta = mid - previous_mid
        previous_mid = mid
        low_total += mid * mid
        high_total += delta * delta
        mid_total += max(0.0, abs(mid) - abs(delta)) ** 2

    frame_count = len(buffer.frames)
    return FrameMetrics(
        rms=sqrt(total_square / sample_count),
        peak=peak,
        low_energy=low_total / frame_count,
        mid_energy=mid_total / frame_count,
        high_energy=high_total / frame_count,
    )


class EnhancementPipeline:
    def __init__(self, profile: EnhancementProfile, sample_rate: int) -> None:
        self.profile = profile
        self.normalizer = LoudnessNormalizer(target_rms=profile.target_rms)
        self.eq = DynamicEq(profile=profile, sample_rate=sample_rate)
        self.limiter = TruePeakLimiter(ceiling=profile.limiter_ceiling)

    def process(self, buffer: AudioFrame, controls: EnhancementControls) -> tuple[AudioFrame, FrameMetrics]:
        metrics = analyze_frame(buffer)
        normalized = self.normalizer.process(buffer, metrics)
        equalized = self.eq.process(normalized, controls=controls)
        limited = self.limiter.process(equalized)
        return limited, analyze_frame(limited)
