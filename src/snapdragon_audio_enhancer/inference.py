from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from enum import Enum

from .audio import AudioBuffer


class BackendKind(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    QNN = "qnn"
    ONNX_QNN = "onnx-qnn"
    WINDOWS_ML = "windows-ml"


@dataclass(frozen=True)
class EnhancementHints:
    clarity: float = 0.08
    warmth: float = 0.04
    stereo_width: float = 0.0


class InferenceBackend:
    """Interface for frame-wise enhancement models."""

    kind = BackendKind.CPU

    def infer_hints(self, frame: AudioBuffer) -> EnhancementHints:
        raise NotImplementedError


class CpuHeuristicBackend(InferenceBackend):
    """Deterministic fallback that mimics model control outputs without dependencies."""

    kind = BackendKind.CPU

    def infer_hints(self, frame: AudioBuffer) -> EnhancementHints:
        peak = frame.peak
        rms = frame.rms
        crest = peak / max(rms, 1.0e-6)
        clarity = 0.04 if crest > 7.0 else 0.09
        warmth = 0.07 if rms < 0.08 else 0.035
        width = 0.03 if frame.channels == 2 and peak < 0.85 else 0.0
        return EnhancementHints(clarity=clarity, warmth=warmth, stereo_width=width)


class OnnxQnnBackend(InferenceBackend):
    """Placeholder adapter for ONNX Runtime QNN Execution Provider on Snapdragon X."""

    kind = BackendKind.ONNX_QNN

    def __init__(self, model_path: str) -> None:
        if not model_path:
            raise ValueError("model_path is required for ONNX QNN inference")
        if importlib.util.find_spec("onnxruntime") is None:
            raise RuntimeError("onnxruntime is not installed")
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        self.model_path = model_path

    def infer_hints(self, frame: AudioBuffer) -> EnhancementHints:
        # The public prototype defines the integration seam while remaining runnable
        # on CI without Qualcomm QNN runtime libraries.
        return CpuHeuristicBackend().infer_hints(frame)


def create_backend(kind: BackendKind | str = BackendKind.CPU, model_path: str | None = None) -> InferenceBackend:
    backend_kind = BackendKind(kind)
    if backend_kind in (BackendKind.AUTO, BackendKind.CPU):
        return CpuHeuristicBackend()
    if backend_kind == BackendKind.QNN:
        return CpuHeuristicBackend()
    if backend_kind == BackendKind.ONNX_QNN:
        try:
            return OnnxQnnBackend(model_path or "")
        except (FileNotFoundError, RuntimeError, ValueError):
            return CpuHeuristicBackend()
    if backend_kind == BackendKind.WINDOWS_ML:
        return CpuHeuristicBackend()
    raise ValueError(f"Unsupported backend: {kind}")


create_inference_backend = create_backend
