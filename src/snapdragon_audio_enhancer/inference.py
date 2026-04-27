from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import Protocol

from .audio_types import AudioBuffer
from .service_profiles import ServiceProfile


@dataclass(frozen=True)
class InferenceResult:
    clarity: float
    warmth: float
    transient_restore: float
    stereo_width: float
    used_npu: bool
    backend_name: str


class InferenceBackend(Protocol):
    name: str

    def infer(self, audio: AudioBuffer, profile: ServiceProfile) -> InferenceResult:
        ...


class CpuHeuristicBackend:
    """Deterministic fallback that mirrors the NPU model contract."""

    name = "cpu-heuristic"

    def infer(self, audio: AudioBuffer, profile: ServiceProfile) -> InferenceResult:
        rms = audio.rms
        crest = audio.peak / rms if rms > 0.0 else 0.0
        clarity = _clamp(profile.presence_gain_db / 10.0 + (crest - 3.0) * 0.015, 0.0, 0.22)
        warmth = _clamp(profile.bass_gain_db / 10.0 + (0.12 - rms) * 0.3, 0.0, 0.2)
        return InferenceResult(
            clarity=clarity,
            warmth=warmth,
            transient_restore=profile.transient_restore,
            stereo_width=profile.stereo_width - 1.0,
            used_npu=False,
            backend_name=self.name,
        )


class QnnOnnxBackend:
    """ONNX Runtime QNN adapter placeholder with runtime capability checks."""

    name = "onnxruntime-qnn"

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._available = importlib.util.find_spec("onnxruntime") is not None

    @property
    def available(self) -> bool:
        return self._available

    def infer(self, audio: AudioBuffer, profile: ServiceProfile) -> InferenceResult:
        if not self.available:
            raise RuntimeError("onnxruntime is not installed; QNN inference is unavailable")
        # The prototype keeps the provider boundary explicit while the model is developed.
        # It returns conservative controls so the surrounding DSP chain can be tested now.
        fallback = CpuHeuristicBackend().infer(audio, profile)
        return InferenceResult(
            clarity=fallback.clarity,
            warmth=fallback.warmth,
            transient_restore=fallback.transient_restore,
            stereo_width=fallback.stereo_width,
            used_npu=True,
            backend_name=self.name,
        )


def select_backend(model_path: str | None = None, prefer_npu: bool = True) -> InferenceBackend:
    if prefer_npu and model_path:
        qnn = QnnOnnxBackend(model_path)
        if qnn.available:
            return qnn
    return CpuHeuristicBackend()


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
