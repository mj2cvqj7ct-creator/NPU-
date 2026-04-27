from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os

from .audio import AudioFrame
from .profiles import EnhancementProfile


class BackendMode(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    NPU = "npu"


@dataclass(frozen=True)
class AudioFeatures:
    rms_db: float
    peak_db: float
    crest_db: float
    zero_crossing_rate: float
    stereo_width: float


@dataclass(frozen=True)
class EnhancementControls:
    loudness_gain_db: float
    bass_gain_db: float
    clarity_gain_db: float
    compression_amount: float
    stereo_width: float


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class FeatureExtractor:
    def analyze(self, frame: AudioFrame) -> AudioFeatures:
        if not frame.samples:
            return AudioFeatures(-120.0, -120.0, 0.0, 0.0, 1.0)

        crossings = 0
        previous = 0.0
        side_energy = 0.0
        mid_energy = 0.0
        for left, right in frame.samples:
            mono = (left + right) * 0.5
            if mono == 0.0:
                sign = previous
            else:
                sign = 1.0 if mono > 0.0 else -1.0
            if previous and sign and sign != previous:
                crossings += 1
            previous = sign
            mid = (left + right) * 0.5
            side = (left - right) * 0.5
            mid_energy += mid * mid
            side_energy += side * side

        zcr = crossings / max(1, len(frame.samples) - 1)
        stereo_width = (side_energy / mid_energy) ** 0.5 if mid_energy > 1e-12 else 1.0
        crest = frame.peak_db - frame.rms_db
        return AudioFeatures(
            rms_db=frame.rms_db,
            peak_db=frame.peak_db,
            crest_db=crest,
            zero_crossing_rate=zcr,
            stereo_width=stereo_width,
        )


class InferenceBackend:
    name = "base"

    def infer(self, frame: AudioFrame, profile: EnhancementProfile) -> tuple[AudioFeatures, EnhancementControls]:
        raise NotImplementedError


class CpuAdaptiveBackend(InferenceBackend):
    """Deterministic control model used until QNN/ONNX NPU inference is available."""

    name = "cpu-adaptive"

    def __init__(self) -> None:
        self._extractor = FeatureExtractor()

    def infer(self, frame: AudioFrame, profile: EnhancementProfile) -> tuple[AudioFeatures, EnhancementControls]:
        features = self._extractor.analyze(frame)
        loudness_error = profile.target_rms_db - features.rms_db
        loudness_gain = _clamp(loudness_error, -profile.max_gain_db, profile.max_gain_db)

        density = _clamp((12.0 - features.crest_db) / 12.0, 0.0, 1.0)
        dullness = _clamp((0.08 - features.zero_crossing_rate) / 0.08, 0.0, 1.0)
        thinness = _clamp((features.zero_crossing_rate - 0.16) / 0.18, 0.0, 1.0)

        bass_gain = profile.bass_boost_db * (0.55 + thinness * 0.45)
        clarity_gain = profile.clarity_boost_db * (0.45 + dullness * 0.55)
        compression = _clamp(profile.compression_strength * (0.4 + density), 0.0, 0.55)
        width = min(profile.stereo_width_limit, max(0.85, features.stereo_width))

        if features.peak_db > profile.limiter_ceiling_db - 1.0:
            loudness_gain = min(loudness_gain, profile.limiter_ceiling_db - features.peak_db)

        return features, EnhancementControls(
            loudness_gain_db=loudness_gain,
            bass_gain_db=bass_gain,
            clarity_gain_db=clarity_gain,
            compression_amount=compression,
            stereo_width=width,
        )


class QualcommQnnBackend(CpuAdaptiveBackend):
    """Placeholder for the Snapdragon X QNN path.

    The class deliberately requires an opt-in environment flag so local runs do not
    pretend to use NPU acceleration before ONNX Runtime QNN or Qualcomm QNN bindings
    are integrated.
    """

    name = "snapdragon-qnn-placeholder"

    @classmethod
    def is_available(cls) -> bool:
        return os.environ.get("SNAPDRAGON_QNN_ENABLED") == "1"


def create_backend(mode: BackendMode | str = BackendMode.AUTO) -> InferenceBackend:
    selected = BackendMode(mode)
    if selected == BackendMode.CPU:
        return CpuAdaptiveBackend()
    if selected == BackendMode.NPU:
        if not QualcommQnnBackend.is_available():
            raise RuntimeError("Snapdragon QNN backend is not configured")
        return QualcommQnnBackend()
    if QualcommQnnBackend.is_available():
        return QualcommQnnBackend()
    return CpuAdaptiveBackend()
