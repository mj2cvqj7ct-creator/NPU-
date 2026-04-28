"""NPU inference abstraction for Snapdragon X audio enhancement.

The production path is expected to use ONNX Runtime with the QNN execution
provider. This module keeps the DSP pipeline usable on non-Snapdragon CI and
developer machines by exposing a deterministic CPU heuristic with the same
contract as a future model runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .audio import AudioBuffer
from .dsp import AudioFeatures, extract_features


class InferenceBackend(str, Enum):
    """Supported inference backends in priority order."""

    QNN = "qnn"
    DIRECTML = "directml"
    CPU = "cpu"


@dataclass(frozen=True)
class EnhancementControls:
    """Continuous DSP controls predicted from local audio features."""

    clarity_db: float
    warmth_db: float
    stereo_width: float
    transient_restore: float
    loudness_target_lufs: float


class AudioEnhancementModel:
    """Contract implemented by NPU-backed and fallback inference models."""

    def infer(self, audio: AudioBuffer) -> EnhancementControls:
        raise NotImplementedError


class HeuristicEnhancementModel(AudioEnhancementModel):
    """CPU fallback that approximates NPU model output from simple features."""

    def infer(self, audio: AudioBuffer) -> EnhancementControls:
        features = extract_features(audio)
        crest = _crest_factor_db(features)
        brightness = features.high_band_energy
        stereo = features.stereo_width

        clarity_db = 0.8
        if brightness < 0.18:
            clarity_db += 1.2
        elif brightness > 0.42:
            clarity_db -= 0.6

        warmth_db = 0.6
        if brightness > 0.36:
            warmth_db += 0.7

        transient_restore = 0.0
        if crest < 9.0:
            transient_restore = min(0.35, (9.0 - crest) / 18.0)

        stereo_width = 1.0
        if stereo > 0.85:
            stereo_width = 1.06
        elif stereo < 0.15:
            stereo_width = 0.94

        return EnhancementControls(
            clarity_db=max(-1.5, min(2.5, clarity_db)),
            warmth_db=max(-1.0, min(2.0, warmth_db)),
            stereo_width=max(0.85, min(1.15, stereo_width)),
            transient_restore=transient_restore,
            loudness_target_lufs=-16.0,
        )


class QnnOnnxEnhancementModel(AudioEnhancementModel):
    """Placeholder for ONNX Runtime QNN EP integration.

    The class intentionally fails fast until model assets and runtime bindings
    are available. Keeping the interface now lets the CLI and pipeline choose
    QNN first on Snapdragon X without changing downstream DSP code later.
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        raise RuntimeError(
            "QNN inference requires ONNX Runtime with QNN Execution Provider "
            "and a calibrated enhancement model."
        )

    def infer(self, audio: AudioBuffer) -> EnhancementControls:
        raise AssertionError("QNN model construction should fail until implemented.")


def create_model(backend: InferenceBackend, model_path: str | None = None) -> AudioEnhancementModel:
    """Create an enhancement model for the requested backend."""

    if backend is InferenceBackend.QNN:
        if not model_path:
            raise ValueError("QNN backend requires --model-path.")
        return QnnOnnxEnhancementModel(model_path)
    if backend is InferenceBackend.DIRECTML:
        raise RuntimeError("DirectML fallback is not implemented in this prototype.")
    return HeuristicEnhancementModel()


def _crest_factor_db(features: AudioFeatures) -> float:
    return max(0.0, features.peak_dbfs - features.rms_dbfs)
