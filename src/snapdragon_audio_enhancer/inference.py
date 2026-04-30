"""Inference backends for Snapdragon X NPU-assisted enhancement.

The production path is expected to use Qualcomm QNN directly or through ONNX
Runtime's QNN execution provider on ARM64 Windows. This module keeps that
integration optional so the DSP pipeline remains testable on development hosts.
"""

from __future__ import annotations

import importlib.util
import platform
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np


class BackendKind(str, Enum):
    QNN = "qnn"
    ONNX_QNN = "onnx-qnn"
    DIRECTML = "directml"
    CPU = "cpu"


@dataclass(frozen=True)
class InferenceRequest:
    samples: np.ndarray
    sample_rate: int
    features: dict[str, float]


class InferenceBackend:
    """Common interface for optional enhancement model backends."""

    kind: BackendKind = BackendKind.CPU

    def enhance(self, request: InferenceRequest) -> np.ndarray:
        raise NotImplementedError


class CpuFallbackBackend(InferenceBackend):
    """Deterministic fallback that applies a tiny clarity curve in NumPy."""

    kind = BackendKind.CPU

    def enhance(self, request: InferenceRequest) -> np.ndarray:
        samples = np.asarray(request.samples, dtype=np.float32)
        if samples.size == 0:
            return samples.copy()

        density = max(0.0, min(1.0, request.features.get("spectral_density", 0.5)))
        clarity_gain = 1.0 + 0.035 * (1.0 - density)
        enhanced = samples.copy()
        enhanced *= clarity_gain
        return np.clip(enhanced, -1.0, 1.0).astype(np.float32)


class OnnxRuntimeBackend(InferenceBackend):
    """ONNX Runtime wrapper, preferring QNN EP when it is available."""

    def __init__(self, model_path: Path, providers: list[str]) -> None:
        import onnxruntime as ort

        self.kind = BackendKind.ONNX_QNN if "QNNExecutionProvider" in providers else BackendKind.DIRECTML
        self._session = ort.InferenceSession(str(model_path), providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    def enhance(self, request: InferenceRequest) -> np.ndarray:
        output = self._session.run(
            [self._output_name],
            {self._input_name: request.samples.astype(np.float32)[None, :, :]},
        )[0]
        return np.asarray(output[0], dtype=np.float32)


@dataclass(frozen=True)
class BackendSelection:
    backend: InferenceBackend
    name: str
    reason: str

    def enhance(self, request: InferenceRequest) -> np.ndarray:
        return self.backend.enhance(request)


def select_backend(model_path: Path | None = None, prefer_npu: bool = True) -> BackendSelection:
    """Choose the best available backend for this host.

    QNN is only selected when ONNX Runtime is installed, a model is provided, and
    the QNN execution provider is exposed. Otherwise the function degrades to
    DirectML or the deterministic CPU fallback.
    """

    if model_path is None or not model_path.exists():
        return BackendSelection(CpuFallbackBackend(), BackendKind.CPU.value, "no ONNX model configured")

    if importlib.util.find_spec("onnxruntime") is None:
        return BackendSelection(CpuFallbackBackend(), BackendKind.CPU.value, "onnxruntime is not installed")

    import onnxruntime as ort

    available = set(ort.get_available_providers())
    machine = platform.machine().lower()
    is_arm64 = machine in {"arm64", "aarch64"}

    if prefer_npu and is_arm64 and "QNNExecutionProvider" in available:
        return BackendSelection(
            OnnxRuntimeBackend(model_path, ["QNNExecutionProvider", "CPUExecutionProvider"]),
            BackendKind.ONNX_QNN.value,
            "selected ONNX Runtime QNN Execution Provider",
        )

    if "DmlExecutionProvider" in available:
        return BackendSelection(
            OnnxRuntimeBackend(model_path, ["DmlExecutionProvider", "CPUExecutionProvider"]),
            BackendKind.DIRECTML.value,
            "selected DirectML fallback provider",
        )

    if "CPUExecutionProvider" in available:
        return BackendSelection(
            OnnxRuntimeBackend(model_path, ["CPUExecutionProvider"]),
            BackendKind.CPU.value,
            "selected ONNX Runtime CPU provider",
        )

    return BackendSelection(CpuFallbackBackend(), BackendKind.CPU.value, "no compatible ONNX Runtime provider")


def build_backend(
    model_path: Path | None = None,
    sample_rate: int = 48_000,
    preference: str = "auto",
) -> BackendSelection:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if preference == "cpu":
        return BackendSelection(CpuFallbackBackend(), BackendKind.CPU.value, "CPU backend requested")
    if preference == "qnn":
        return select_backend(model_path=model_path, prefer_npu=True)
    if preference == "auto":
        return select_backend(model_path=model_path, prefer_npu=True)
    raise ValueError(f"Unsupported backend preference: {preference}")
