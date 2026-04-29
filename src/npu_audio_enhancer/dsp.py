from __future__ import annotations

from dataclasses import dataclass
from math import log10, pi, sin

import numpy as np

from .audio import AudioBuffer
from .profiles import EnhancementProfile


@dataclass(frozen=True)
class EnhancementMetrics:
    peak_before: float
    peak_after: float
    rms_before: float
    rms_after: float
    applied_gain_db: float
    limiter_reductions: int


def rms(buffer: AudioBuffer) -> float:
    if buffer.frames == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(buffer.samples, dtype=np.float32))))


def peak(buffer: AudioBuffer) -> float:
    if buffer.frames == 0:
        return 0.0
    return float(np.max(np.abs(buffer.samples)))


class DynamicEq:
    """Small three-band EQ tuned from streaming-service and listener context."""

    def __init__(self, sample_rate: int, low_gain: float, presence_gain: float, air_gain: float) -> None:
        self.low_alpha = _one_pole_alpha(160.0, sample_rate)
        self.presence_alpha = _one_pole_alpha(2_800.0, sample_rate)
        self.low_gain = low_gain
        self.presence_gain = presence_gain
        self.air_gain = air_gain
        self.low_state = 0.0
        self.presence_state = 0.0

    def process_sample(self, sample: float) -> float:
        self.low_state += self.low_alpha * (sample - self.low_state)
        low = self.low_state
        high_input = sample - low
        self.presence_state += self.presence_alpha * (high_input - self.presence_state)
        presence = self.presence_state
        air = high_input - presence
        return sample + low * self.low_gain + presence * self.presence_gain + air * self.air_gain


class SoftKneeLimiter:
    def __init__(self, threshold: float = 0.94, ceiling: float = 0.985) -> None:
        self.threshold = threshold
        self.ceiling = ceiling
        self.reductions = 0

    def process_sample(self, sample: float) -> float:
        magnitude = abs(sample)
        if magnitude <= self.threshold:
            return sample

        self.reductions += 1
        sign = 1.0 if sample >= 0.0 else -1.0
        excess = magnitude - self.threshold
        compressed = self.threshold + (self.ceiling - self.threshold) * sin(
            min(excess / (1.0 - self.threshold), 1.0) * pi / 2.0
        )
        return sign * min(compressed, self.ceiling)


class DspEnhancer:
    def __init__(self, profile: EnhancementProfile) -> None:
        self.profile = profile

    def process(self, buffer: AudioBuffer, npu_features: dict[str, float] | None = None) -> tuple[AudioBuffer, EnhancementMetrics]:
        if buffer.channels != 2:
            raise ValueError("enhancement pipeline expects stereo PCM")

        before_peak = peak(buffer)
        before_rms = rms(buffer)
        gain_db = self._target_gain_db(before_rms, npu_features or {})
        linear_gain = 10 ** (gain_db / 20.0)
        clarity = (npu_features or {}).get("clarity", 0.5)
        density = (npu_features or {}).get("density", 0.5)

        eq = DynamicEq(
            sample_rate=buffer.sample_rate_hz,
            low_gain=self.profile.low_shelf + (0.025 if density < 0.45 else -0.015),
            presence_gain=self.profile.presence_boost + (clarity - 0.5) * 0.08,
            air_gain=self.profile.air_boost,
        )
        limiter = SoftKneeLimiter()
        processed = np.empty_like(buffer.samples, dtype=np.float32)

        for frame_index, frame in enumerate(buffer.samples):
            for channel_index, sample in enumerate(frame):
                enhanced = eq.process_sample(sample * linear_gain)
                processed[frame_index, channel_index] = limiter.process_sample(float(enhanced))

        processed_audio = AudioBuffer(samples=processed, sample_rate_hz=buffer.sample_rate_hz)
        return processed_audio, EnhancementMetrics(
            peak_before=before_peak,
            peak_after=peak(processed_audio),
            rms_before=before_rms,
            rms_after=rms(processed_audio),
            applied_gain_db=gain_db,
            limiter_reductions=limiter.reductions,
        )

    def _target_gain_db(self, current_rms: float, npu_features: dict[str, float]) -> float:
        if current_rms <= 0.000_001:
            return 0.0
        target = self.profile.target_rms
        if npu_features.get("transient_risk", 0.0) > 0.75:
            target *= 0.88
        gain = 20.0 * log10(target / current_rms)
        return max(min(gain, self.profile.max_gain_db), -8.0)


def _one_pole_alpha(cutoff_hz: float, sample_rate: int) -> float:
    return min(1.0, max(0.0, 2.0 * pi * cutoff_hz / (2.0 * pi * cutoff_hz + sample_rate)))
