"""NPU inference boundary for Snapdragon X audio enhancement.

The production implementation should bind this interface to Qualcomm QNN or
ONNX Runtime QNN Execution Provider. The reference model is deterministic and
small enough for tests while preserving the same frame-level contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .dsp import AudioFeatures


@dataclass(frozen=True)
class EnhancementControls:
    """Frame-local controls normally predicted by an NPU model."""

    clarity: float
    warmth: float
    stereo_width: float
    transient_restore: float
    loudness_boost_db: float


class NpuEnhancementModel(Protocol):
    """Protocol implemented by QNN/ONNX Runtime backed models."""

    def infer(self, features: AudioFeatures) -> EnhancementControls:
        """Return enhancement controls for one audio frame."""


class HeuristicNpuModel:
    """CPU reference for the Snapdragon X NPU control model.

    This intentionally produces conservative controls. It mirrors the kind of
    low-dimensional parameters a tiny NPU model would emit, without pretending
    to reconstruct information lost by lossy streaming codecs.
    """

    def infer(self, features: AudioFeatures) -> EnhancementControls:
        density = _clamp(features.spectral_density, 0.0, 1.0)
        low_energy = _clamp(1.0 - features.rms / 0.18, 0.0, 1.0)
        clipped_penalty = 1.0 - _clamp(features.clipping_ratio * 20.0, 0.0, 0.8)

        clarity = _clamp(0.18 + density * 0.22, 0.0, 0.42) * clipped_penalty
        warmth = _clamp(0.12 + (1.0 - density) * 0.18, 0.0, 0.35)
        stereo_width = _clamp(1.0 + density * 0.08 - features.channel_imbalance * 0.08, 0.92, 1.08)
        transient_restore = _clamp((features.crest_factor - 2.0) / 8.0, 0.0, 0.28)
        loudness_boost_db = _clamp(low_energy * 2.5, 0.0, 2.5)

        return EnhancementControls(
            clarity=clarity,
            warmth=warmth,
            stereo_width=stereo_width,
            transient_restore=transient_restore,
            loudness_boost_db=loudness_boost_db,
        )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
