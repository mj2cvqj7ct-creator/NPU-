from __future__ import annotations

import math
from dataclasses import dataclass

from .profiles import EnhancementProfile


@dataclass(frozen=True)
class EnhancementFeatures:
    clarity: float
    bass_tightness: float
    transient_restore: float
    stereo_focus: float
    noise_floor: float
    vocal_presence: float

    @classmethod
    def zero(cls) -> "EnhancementFeatures":
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    @classmethod
    def neutral(cls) -> "EnhancementFeatures":
        return cls(0.5, 0.5, 0.0, 0.5, 0.0, 0.5)

    def __add__(self, other: "EnhancementFeatures") -> "EnhancementFeatures":
        return EnhancementFeatures(
            clarity=self.clarity + other.clarity,
            bass_tightness=self.bass_tightness + other.bass_tightness,
            transient_restore=self.transient_restore + other.transient_restore,
            stereo_focus=self.stereo_focus + other.stereo_focus,
            noise_floor=self.noise_floor + other.noise_floor,
            vocal_presence=self.vocal_presence + other.vocal_presence,
        )

    def scale(self, factor: float) -> "EnhancementFeatures":
        return EnhancementFeatures(
            clarity=self.clarity * factor,
            bass_tightness=self.bass_tightness * factor,
            transient_restore=self.transient_restore * factor,
            stereo_focus=self.stereo_focus * factor,
            noise_floor=self.noise_floor * factor,
            vocal_presence=self.vocal_presence * factor,
        )


NpuFeatures = EnhancementFeatures


class HeuristicNpuModel:
    """CPU-verifiable stand-in for the future ONNX Runtime QNN model."""

    backend_name = "cpu-heuristic-qnn-compatible"

    def infer(self, samples: list[float], profile: EnhancementProfile) -> EnhancementFeatures:
        if not samples:
            return EnhancementFeatures.neutral()

        mono = _mono(samples)
        rms = math.sqrt(sum(sample * sample for sample in mono) / len(mono))
        peak = max(abs(sample) for sample in mono)
        crest = peak / max(rms, 1e-9)
        low = _band_energy(mono, 2)
        mid = _band_energy(mono, 8)
        high = _band_energy(mono, 32)
        noise = _noise_floor(mono, rms)
        side = _stereo_width(samples)

        dense_mix = _clamp((mid - high) * 2.2, 0.0, 1.0)
        dull = _clamp(0.30 - high, 0.0, 1.0)
        loose_bass = _clamp(low - 0.38, 0.0, 1.0)
        crest_need = _clamp((crest - 1.7) / 4.2, 0.0, 1.0)

        return EnhancementFeatures(
            clarity=_clamp(0.45 + profile.clarity_weight + dense_mix * 0.32 + dull * 0.16, 0.0, 1.0),
            bass_tightness=_clamp(0.52 + profile.bass_weight - loose_bass * 0.28, 0.0, 1.0),
            transient_restore=_clamp(profile.transient_weight + crest_need * 0.55, 0.0, 1.0),
            stereo_focus=_clamp(0.52 + (1.0 - side) * 0.22, 0.0, 1.0),
            noise_floor=noise,
            vocal_presence=_clamp(0.45 + profile.vocal_focus * 0.18 + dense_mix * 0.28, 0.0, 1.0),
        )


NpuFeatureExtractor = HeuristicNpuModel


def _mono(samples: list[float]) -> list[float]:
    if len(samples) < 2:
        return samples
    return [(samples[index] + samples[index + 1]) * 0.5 for index in range(0, len(samples) - 1, 2)]


def _band_energy(samples: list[float], stride: int) -> float:
    if len(samples) <= stride:
        return 0.0

    total = 0.0
    for index in range(stride, len(samples)):
        diff = samples[index] - samples[index - stride]
        total += diff * diff
    return _clamp(math.sqrt(total / (len(samples) - stride)) * (stride ** 0.5), 0.0, 1.0)


def _noise_floor(samples: list[float], rms: float) -> float:
    zero_crossings = 0
    for index in range(1, len(samples)):
        if (samples[index - 1] < 0 <= samples[index]) or (samples[index - 1] >= 0 > samples[index]):
            zero_crossings += 1
    crossing_rate = zero_crossings / max(1, len(samples) - 1)
    return _clamp((0.12 - rms) * 3.0 + crossing_rate * 0.9, 0.0, 1.0)


def _stereo_width(samples: list[float]) -> float:
    if len(samples) < 2:
        return 0.0

    frames = len(samples) // 2
    mid_total = 0.0
    side_total = 0.0
    for index in range(0, frames * 2, 2):
        mid = (samples[index] + samples[index + 1]) * 0.5
        side = (samples[index] - samples[index + 1]) * 0.5
        mid_total += mid * mid
        side_total += side * side
    mid_rms = math.sqrt(mid_total / frames)
    side_rms = math.sqrt(side_total / frames)
    return _clamp(side_rms / max(mid_rms, 1e-9), 0.0, 1.0)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
