"""Inference backend selection for Snapdragon X oriented audio processing."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from typing import Iterable, Protocol

from .frames import AudioFrame


class InferenceBackend(Protocol):
    """Small interface shared by hardware and fallback inference backends."""

    @property
    def name(self) -> str:
        """Human readable backend name."""

    def process(self, frame: AudioFrame) -> AudioFrame:
        """Return an enhanced audio frame."""


@dataclass(frozen=True)
class BackendAvailability:
    """Explains why a backend was or was not selected."""

    name: str
    available: bool
    reason: str


class CpuFallbackBackend:
    """Deterministic fallback used when no accelerator is available.

    It intentionally performs no model-based enhancement. The rule-based DSP
    chain remains active, while NPU-specific quality models can be added behind
    the same interface without changing capture or rendering code.
    """

    name = "cpu-fallback"

    def process(self, frame: AudioFrame) -> AudioFrame:
        return frame


class QnnNpuBackend:
    """Placeholder boundary for ONNX Runtime QNN Execution Provider support."""

    name = "qnn-npu"

    def __init__(self) -> None:
        if importlib.util.find_spec("onnxruntime") is None:
            raise RuntimeError("onnxruntime is not installed")
        providers = _available_onnx_providers()
        if "QNNExecutionProvider" not in providers:
            raise RuntimeError("ONNX Runtime QNNExecutionProvider is unavailable")

    def process(self, frame: AudioFrame) -> AudioFrame:
        # Future model invocation belongs here. Keeping the pass-through
        # contract makes the audio path safe before model weights exist.
        return frame


class DirectMlBackend:
    """Placeholder boundary for Windows DirectML fallback support."""

    name = "directml"

    def __init__(self) -> None:
        providers = _available_onnx_providers()
        if "DmlExecutionProvider" not in providers:
            raise RuntimeError("ONNX Runtime DmlExecutionProvider is unavailable")

    def process(self, frame: AudioFrame) -> AudioFrame:
        return frame


def select_backend(preferred: Iterable[str] | None = None) -> InferenceBackend:
    """Select the best available inference backend.

    Priority is Snapdragon X NPU via QNN, then DirectML, then a CPU-safe
    pass-through backend. The ``SNAPDRAGON_NPU_AUDIO_BACKEND`` environment
    variable can force ``qnn``, ``directml``, or ``cpu`` for diagnostics.
    """

    forced = os.environ.get("SNAPDRAGON_NPU_AUDIO_BACKEND")
    order = list(preferred or ("qnn", "directml", "cpu"))
    if forced:
        order = [forced.lower()]

    errors: list[str] = []
    for name in order:
        normalized = name.lower()
        try:
            if normalized in {"qnn", "qnn-npu", "npu"}:
                return QnnNpuBackend()
            if normalized in {"directml", "dml"}:
                return DirectMlBackend()
            if normalized in {"cpu", "cpu-fallback"}:
                return CpuFallbackBackend()
        except RuntimeError as exc:
            errors.append(f"{normalized}: {exc}")

    if forced and errors:
        raise RuntimeError(f"Forced backend is unavailable: {'; '.join(errors)}")
    return CpuFallbackBackend()


def probe_backends() -> list[BackendAvailability]:
    """Return backend availability without selecting one."""

    results: list[BackendAvailability] = []
    for label, factory in (
        ("qnn-npu", QnnNpuBackend),
        ("directml", DirectMlBackend),
        ("cpu-fallback", CpuFallbackBackend),
    ):
        try:
            factory()
        except RuntimeError as exc:
            results.append(BackendAvailability(label, False, str(exc)))
        else:
            results.append(BackendAvailability(label, True, "available"))
    return results


def _available_onnx_providers() -> list[str]:
    if importlib.util.find_spec("onnxruntime") is None:
        return []
    import onnxruntime as ort  # type: ignore[import-not-found]

    return list(ort.get_available_providers())
