"""Inference backend abstractions for Snapdragon X audio enhancement."""

from __future__ import annotations

import importlib.util
import platform
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Protocol

from .audio_frame import AudioFrame


class BackendKind(str, Enum):
    """Supported inference backend families in preference order."""

    QNN_ONNX = "qnn_onnx"
    DIRECTML = "directml"
    CPU = "cpu"


@dataclass(frozen=True)
class InferenceConfig:
    """Runtime configuration for model-backed enhancement."""

    preferred_backends: tuple[BackendKind, ...] = (
        BackendKind.QNN_ONNX,
        BackendKind.DIRECTML,
        BackendKind.CPU,
    )
    model_path: str | None = None
    enable_neural_enhancement: bool = True


@dataclass(frozen=True)
class MusicFeatures:
    """Compact features that guide local enhancement decisions."""

    brightness: float
    bass_weight: float
    density: float
    vocal_presence: float
    transient_softness: float


class InferenceBackend(Protocol):
    """Protocol implemented by NPU and fallback inference backends."""

    kind: BackendKind

    def extract_features(self, frame: AudioFrame) -> MusicFeatures:
        """Return local audio features for the current frame."""

    def enhance(self, frame: AudioFrame, features: MusicFeatures) -> AudioFrame:
        """Apply model-like enhancement to the current frame."""


class CpuFeatureBackend:
    """Deterministic fallback used when the Snapdragon NPU is unavailable."""

    kind = BackendKind.CPU

    def extract_features(self, frame: AudioFrame) -> MusicFeatures:
        mono = frame.mono()
        if not mono:
            return MusicFeatures(0.0, 0.0, 0.0, 0.0, 0.0)

        crossings = 0
        prev = mono[0]
        for sample in mono[1:]:
            if (prev <= 0.0 < sample) or (prev >= 0.0 > sample):
                crossings += 1
            prev = sample

        abs_samples = [abs(sample) for sample in mono]
        peak = max(abs_samples)
        mean_abs = sum(abs_samples) / len(abs_samples)
        rms = frame.rms
        crest_factor = peak / max(rms, 1e-6)

        low_energy = _band_energy_proxy(mono, window=32)
        high_energy = crossings / max(1, len(mono) - 1)
        density = min(1.0, rms * 3.0)

        return MusicFeatures(
            brightness=_clamp(high_energy * 6.0, 0.0, 1.0),
            bass_weight=_clamp(low_energy * 4.0, 0.0, 1.0),
            density=_clamp(density, 0.0, 1.0),
            vocal_presence=_clamp(mean_abs * 4.5, 0.0, 1.0),
            transient_softness=_clamp(1.0 / max(crest_factor / 4.0, 1e-6), 0.0, 1.0),
        )

    def enhance(self, frame: AudioFrame, features: MusicFeatures) -> AudioFrame:
        """Apply a conservative neural-enhancement approximation.

        Real deployments should replace this with a quantized ONNX model using
        the QNN execution provider. The fallback deliberately avoids aggressive
        synthesis so that it is safe for real-time auditioning.
        """

        clarity = 1.0 + 0.035 * (1.0 - features.brightness) * features.vocal_presence
        body = 1.0 + 0.025 * (1.0 - features.bass_weight)
        restored: list[tuple[float, float]] = []

        last_left = 0.0
        last_right = 0.0
        for left, right in frame.samples:
            left_delta = left - last_left
            right_delta = right - last_right
            last_left, last_right = left, right

            enhanced_left = left * body + left_delta * clarity * 0.018
            enhanced_right = right * body + right_delta * clarity * 0.018
            restored.append((enhanced_left, enhanced_right))

        return frame.with_samples(restored)


class OnnxQnnBackend(CpuFeatureBackend):
    """Placeholder adapter for ONNX Runtime QNN execution provider."""

    kind = BackendKind.QNN_ONNX


class DirectMlBackend(CpuFeatureBackend):
    """Placeholder adapter for DirectML fallback."""

    kind = BackendKind.DIRECTML


def select_backend(config: InferenceConfig) -> InferenceBackend:
    """Select the best available inference backend for the current machine."""

    for backend in config.preferred_backends:
        if backend is BackendKind.QNN_ONNX and _qnn_onnx_available(config):
            return OnnxQnnBackend()
        if backend is BackendKind.DIRECTML and _directml_available():
            return DirectMlBackend()
        if backend is BackendKind.CPU:
            return CpuFeatureBackend()
    return CpuFeatureBackend()


def available_backend_kinds(config: InferenceConfig | None = None) -> tuple[BackendKind, ...]:
    """Return backends available in this runtime."""

    config = config or InferenceConfig()
    available: list[BackendKind] = []
    if _qnn_onnx_available(config):
        available.append(BackendKind.QNN_ONNX)
    if _directml_available():
        available.append(BackendKind.DIRECTML)
    available.append(BackendKind.CPU)
    return tuple(available)


def _qnn_onnx_available(config: InferenceConfig) -> bool:
    if not config.enable_neural_enhancement or not config.model_path:
        return False
    if platform.machine().lower() not in {"arm64", "aarch64"}:
        return False
    if importlib.util.find_spec("onnxruntime") is None:
        return False

    try:
        import onnxruntime as ort  # type: ignore[import-not-found]
    except Exception:
        return False

    providers: Iterable[str] = ort.get_available_providers()
    return "QNNExecutionProvider" in providers


def _directml_available() -> bool:
    if importlib.util.find_spec("onnxruntime") is None:
        return False
    try:
        import onnxruntime as ort  # type: ignore[import-not-found]
    except Exception:
        return False
    return "DmlExecutionProvider" in ort.get_available_providers()


def _band_energy_proxy(samples: list[float], window: int) -> float:
    if len(samples) < window:
        return 0.0

    totals: list[float] = []
    for start in range(0, len(samples) - window + 1, window):
        chunk = samples[start : start + window]
        totals.append(abs(sum(chunk) / window))
    return sum(totals) / max(1, len(totals))


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))
