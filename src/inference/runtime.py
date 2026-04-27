"""NPU runtime selection for Snapdragon X audio enhancement.

The production target is Qualcomm QNN on Windows ARM64. This module keeps the
runtime contract small enough to test without vendor SDKs while preserving the
backend priority that should be used on device.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import platform
from typing import Protocol, Sequence

from src.dsp.enhancer import NpuEnhancementControls, StereoFrame


class EnhancementRuntime(Protocol):
    """Inference backend that converts PCM frames into DSP control signals."""

    name: str

    def infer_controls(self, frames: Sequence[StereoFrame]) -> NpuEnhancementControls:
        """Infer bounded enhancement controls for one low-latency audio block."""


@dataclass(frozen=True)
class RuntimeCapabilities:
    architecture: str
    system: str
    has_onnxruntime: bool
    prefer_qnn: bool


class CpuFallbackRuntime:
    """Deterministic heuristic fallback when no NPU-capable backend is present."""

    name = "cpu-fallback"

    def infer_controls(self, frames: Sequence[StereoFrame]) -> NpuEnhancementControls:
        if not frames:
            return NpuEnhancementControls()

        mono = [(left + right) * 0.5 for left, right in frames]
        energy = sum(sample * sample for sample in mono) / len(mono)
        zero_crossings = sum(
            1
            for previous, current in zip(mono, mono[1:])
            if (previous < 0.0 <= current) or (previous >= 0.0 > current)
        ) / max(1, len(mono) - 1)
        peak = max(abs(sample) for sample in mono)
        crest = peak / (energy**0.5 + 1e-9)

        return NpuEnhancementControls(
            clarity=min(0.35, zero_crossings * 1.2),
            warmth=0.2 if energy < 0.01 else 0.0,
            de_mud=min(0.45, max(0.0, energy * 4.0 - zero_crossings)),
            transient_restore=min(0.5, max(0.0, crest - 1.5) / 4.0),
            stereo_focus=0.0,
        ).bounded()


class OnnxQnnRuntime:
    """Placeholder adapter for ONNX Runtime QNN Execution Provider integration."""

    name = "onnx-qnn"

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._session = None

    def infer_controls(self, frames: Sequence[StereoFrame]) -> NpuEnhancementControls:
        raise RuntimeError(
            "ONNX Runtime QNN integration is not initialized in this scaffold. "
            "Install onnxruntime-qnn on Windows ARM64 and wire model tensors to "
            "return NpuEnhancementControls."
        )


def detect_capabilities() -> RuntimeCapabilities:
    architecture = platform.machine().lower()
    system = platform.system().lower()
    has_onnxruntime = importlib.util.find_spec("onnxruntime") is not None
    prefer_qnn = system == "windows" and architecture in {"arm64", "aarch64"} and has_onnxruntime
    return RuntimeCapabilities(
        architecture=architecture,
        system=system,
        has_onnxruntime=has_onnxruntime,
        prefer_qnn=prefer_qnn,
    )


def select_runtime(model_path: str | None = None) -> EnhancementRuntime:
    """Select the best available runtime in Snapdragon X priority order."""

    capabilities = detect_capabilities()
    if capabilities.prefer_qnn and model_path:
        return OnnxQnnRuntime(model_path)
    return CpuFallbackRuntime()
