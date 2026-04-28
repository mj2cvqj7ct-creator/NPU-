from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Sequence

StereoFrame = tuple[float, float]
AudioFrame = list[StereoFrame]


def clamp_sample(value: float, limit: float = 1.0) -> float:
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


def rms(samples: Sequence[StereoFrame]) -> float:
    if not samples:
        return 0.0
    total = 0.0
    for left, right in samples:
        total += left * left + right * right
    return sqrt(total / (len(samples) * 2))


def peak(samples: Sequence[StereoFrame]) -> float:
    if not samples:
        return 0.0
    return max(max(abs(left), abs(right)) for left, right in samples)


@dataclass(frozen=True)
class AudioFeatures:
    """Small frame-level feature vector suitable for low-latency routing."""

    rms: float
    peak: float
    crest_factor: float
    stereo_width: float
    clipping_ratio: float


class FeatureExtractor:
    def analyze(self, samples: Sequence[StereoFrame]) -> AudioFeatures:
        level = rms(samples)
        frame_peak = peak(samples)
        clipped = 0
        side_energy = 0.0
        mid_energy = 0.0

        for left, right in samples:
            if abs(left) >= 0.999 or abs(right) >= 0.999:
                clipped += 1
            mid = (left + right) * 0.5
            side = (left - right) * 0.5
            mid_energy += mid * mid
            side_energy += side * side

        crest = frame_peak / level if level > 1e-9 else 0.0
        width = sqrt(side_energy / mid_energy) if mid_energy > 1e-9 else 0.0
        clipping_ratio = clipped / len(samples) if samples else 0.0
        return AudioFeatures(level, frame_peak, crest, width, clipping_ratio)


@dataclass(frozen=True)
class EnhancementProfile:
    name: str
    target_rms: float
    max_gain: float
    bass_tilt: float
    presence_tilt: float
    stereo_width: float
    limiter_ceiling: float = 0.98


SERVICE_PROFILES: dict[str, EnhancementProfile] = {
    "spotify": EnhancementProfile(
        name="spotify",
        target_rms=0.18,
        max_gain=1.8,
        bass_tilt=0.02,
        presence_tilt=0.035,
        stereo_width=1.04,
    ),
    "apple_music": EnhancementProfile(
        name="apple_music",
        target_rms=0.16,
        max_gain=1.6,
        bass_tilt=0.01,
        presence_tilt=0.02,
        stereo_width=1.02,
    ),
    "youtube_music": EnhancementProfile(
        name="youtube_music",
        target_rms=0.17,
        max_gain=1.9,
        bass_tilt=0.015,
        presence_tilt=0.03,
        stereo_width=1.03,
    ),
    "generic": EnhancementProfile(
        name="generic",
        target_rms=0.16,
        max_gain=1.7,
        bass_tilt=0.015,
        presence_tilt=0.025,
        stereo_width=1.02,
    ),
}


@dataclass(frozen=True)
class EnhancementDecision:
    gain: float
    bass_tilt: float
    presence_tilt: float
    stereo_width: float


class RuleBasedEnhancer:
    """Deterministic real-time DSP fallback used before and after NPU inference."""

    def decide(self, features: AudioFeatures, profile: EnhancementProfile) -> EnhancementDecision:
        if features.rms <= 1e-9:
            gain = 1.0
        else:
            gain = min(profile.max_gain, profile.target_rms / features.rms)

        # Already dense or clipped masters should be protected from aggressive boosts.
        if features.crest_factor < 4.0 or features.clipping_ratio > 0.01:
            gain = min(gain, 1.05)
            presence = profile.presence_tilt * 0.4
        else:
            presence = profile.presence_tilt

        width = min(profile.stereo_width, 1.0 + max(0.0, 0.7 - features.stereo_width) * 0.08)
        return EnhancementDecision(
            gain=gain,
            bass_tilt=profile.bass_tilt,
            presence_tilt=presence,
            stereo_width=width,
        )

    def process(
        self,
        samples: Sequence[StereoFrame],
        profile: EnhancementProfile,
        decision: EnhancementDecision | None = None,
    ) -> list[StereoFrame]:
        if not samples:
            return []

        features = FeatureExtractor().analyze(samples)
        decision = decision or self.decide(features, profile)
        processed: list[StereoFrame] = []
        low_state_l = 0.0
        low_state_r = 0.0
        prev_l = samples[0][0]
        prev_r = samples[0][1]

        for left, right in samples:
            low_state_l = low_state_l * 0.985 + left * 0.015
            low_state_r = low_state_r * 0.985 + right * 0.015
            transient_l = left - prev_l
            transient_r = right - prev_r

            shaped_l = left * decision.gain
            shaped_r = right * decision.gain
            shaped_l += low_state_l * decision.bass_tilt + transient_l * decision.presence_tilt
            shaped_r += low_state_r * decision.bass_tilt + transient_r * decision.presence_tilt

            mid = (shaped_l + shaped_r) * 0.5
            side = (shaped_l - shaped_r) * 0.5 * decision.stereo_width
            processed.append((mid + side, mid - side))
            prev_l, prev_r = left, right

        return TruePeakLimiter(profile.limiter_ceiling).process(processed)


class TruePeakLimiter:
    def __init__(self, ceiling: float = 0.98) -> None:
        self.ceiling = ceiling

    def process(self, samples: Iterable[StereoFrame]) -> list[StereoFrame]:
        output: list[StereoFrame] = []
        for left, right in samples:
            frame_peak = max(abs(left), abs(right))
            if frame_peak > self.ceiling:
                scale = self.ceiling / frame_peak
                left *= scale
                right *= scale
            output.append((clamp_sample(left, self.ceiling), clamp_sample(right, self.ceiling)))
        return output
