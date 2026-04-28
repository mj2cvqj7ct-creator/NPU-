"""Backend selection and inference facade for Snapdragon X NPU acceleration.

The production implementation should load a quantized ONNX model through the
ONNX Runtime QNN Execution Provider or Qualcomm's native QNN SDK. This module
keeps that integration behind a small interface so the real-time audio path can
fall back cleanly when those providers are unavailable on the current machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import importlib.util
import os

from .dsp import AudioFeatures, StereoFrame


class BackendKind(str, Enum):
    QNN = "qnn"
    DIRECTML = "directml"
    CPU = "cpu"


@dataclass(frozen=True)
class InferenceResult:
    """Small control signal emitted by the model or fallback heuristic."""

    clarity: float
    warmth: float
    spatial: float
    transient: float


@dataclass(frozen=True)
class BackendInfo:
    kind: BackendKind
    provider: str
    accelerated: bool
    reason: str


class NpuEnhancer:
    """Selects the best available inference backend and emits DSP controls."""

    def __init__(self, model_path: str | None = None, preferred: BackendKind | None = None) -> None:
        self.model_path = model_path
        self.backend = self._select_backend(preferred)

    def infer(self, frame: list[StereoFrame], features: AudioFeatures) -> InferenceResult:
        """Return enhancement controls for one frame.

        Until a trained model is available, the CPU path uses bounded heuristics
        that mirror the intended model outputs. This lets the rest of the audio
        pipeline be tested without pretending to alter service algorithms.
        """

        del frame
        density = min(1.0, features.rms * 3.0)
        clipping_penalty = 0.08 if features.peak > 0.98 else 0.0
        openness = _clamp(features.crest_factor / 12.0, 0.0, 1.0)
        clarity = _clamp(0.08 + openness * 0.16 - clipping_penalty, 0.0, 0.24)
        warmth = _clamp(0.08 + (1.0 - density) * 0.12, 0.0, 0.20)
        spatial = _clamp(0.04 + min(features.stereo_width, 1.0) * 0.12, 0.0, 0.16)
        transient = _clamp(0.06 + density * 0.14 - clipping_penalty, 0.0, 0.20)
        return InferenceResult(clarity=clarity, warmth=warmth, spatial=spatial, transient=transient)

    def _select_backend(self, preferred: BackendKind | None) -> BackendInfo:
        forced = os.getenv("NPU_AUDIO_BACKEND")
        requested = preferred or (BackendKind(forced.lower()) if forced else None)
        available = _available_ort_providers()

        if requested is BackendKind.QNN or requested is None:
            if "QNNExecutionProvider" in available:
                return BackendInfo(BackendKind.QNN, "QNNExecutionProvider", True, "ONNX Runtime QNN provider available")
            if requested is BackendKind.QNN:
                return BackendInfo(BackendKind.CPU, "CPUExecutionProvider", False, "QNN provider unavailable")

        if requested is BackendKind.DIRECTML or requested is None:
            if "DmlExecutionProvider" in available:
                return BackendInfo(BackendKind.DIRECTML, "DmlExecutionProvider", True, "DirectML provider available")
            if requested is BackendKind.DIRECTML:
                return BackendInfo(BackendKind.CPU, "CPUExecutionProvider", False, "DirectML provider unavailable")

        return BackendInfo(BackendKind.CPU, "CPUExecutionProvider", False, "accelerated provider unavailable")


def _available_ort_providers() -> set[str]:
    if importlib.util.find_spec("onnxruntime") is None:
        return set()
    import onnxruntime as ort  # type: ignore[import-not-found]

    return set(ort.get_available_providers())


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
