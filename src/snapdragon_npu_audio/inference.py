"""Inference boundary for Snapdragon X NPU backed enhancement decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .profiles import ServiceProfile


@dataclass(frozen=True)
class AudioFeatures:
    """Short-window audio descriptors used to steer enhancement."""

    rms_db: float
    peak_db: float
    low_band_energy: float
    mid_band_energy: float
    high_band_energy: float
    transient_score: float
    sample_rate: int


@dataclass(frozen=True)
class EnhancementPlan:
    """Per-frame control values that the DSP stage can apply cheaply."""

    loudness_gain_db: float
    low_shelf_db: float
    presence_gain_db: float
    compression_ratio: float
    transient_restore: float
    stereo_width: float


class EnhancementBackend(Protocol):
    """Backend contract implemented by QNN, ONNX Runtime, or fallback code."""

    name: str

    def infer(self, features: AudioFeatures, profile: ServiceProfile) -> EnhancementPlan:
        """Return frame-level enhancement controls."""


class HeuristicNpuBackend:
    """Deterministic backend shaped like the intended NPU model contract.

    The real Snapdragon X path can replace this class with a QNN/ONNX Runtime
    implementation while keeping the DSP stage and tests unchanged.
    """

    name = "heuristic-npu-contract"

    def infer(self, features: AudioFeatures, profile: ServiceProfile) -> EnhancementPlan:
        loudness_gap = profile.target_loudness_db - features.rms_db
        loudness_gain_db = _clamp(loudness_gap * 0.42, -6.0, 8.0)

        low_shelf_db = profile.bass_tilt_db
        if features.low_band_energy < 0.72:
            low_shelf_db += 1.2
        elif features.low_band_energy > 1.35:
            low_shelf_db -= 1.0

        presence_gain_db = profile.presence_db
        if features.mid_band_energy < 0.82:
            presence_gain_db += 0.9

        if features.high_band_energy < 0.65:
            presence_gain_db += 0.4
        elif features.high_band_energy > 1.45:
            presence_gain_db -= 0.5

        compression_ratio = profile.compression_ratio
        if features.peak_db - features.rms_db > 17.0:
            compression_ratio = max(1.0, compression_ratio - 0.15)
        elif features.rms_db > -13.0:
            compression_ratio = min(2.6, compression_ratio + 0.25)

        transient_restore = profile.transient_restore
        if features.transient_score > 0.42:
            transient_restore *= 0.55
        elif features.transient_score < 0.16:
            transient_restore += 0.12

        return EnhancementPlan(
            loudness_gain_db=loudness_gain_db,
            low_shelf_db=_clamp(low_shelf_db, -3.0, 4.0),
            presence_gain_db=_clamp(presence_gain_db, -2.0, 3.5),
            compression_ratio=_clamp(compression_ratio, 1.0, 3.0),
            transient_restore=_clamp(transient_restore, 0.0, 0.65),
            stereo_width=_clamp(profile.stereo_width, 0.85, 1.18),
        )


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
