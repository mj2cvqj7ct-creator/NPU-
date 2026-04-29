"""Inference backend selection for Snapdragon X NPU assisted enhancement."""

from __future__ import annotations

import importlib.util
import os
import platform
from dataclasses import dataclass
from enum import Enum

from .models import AudioFrame, EnhancementSettings, FrameFeatures, InferenceDecision, ServiceProfile


class BackendKind(str, Enum):
    """Supported inference execution targets in preference order."""

    QNN_NPU = "qnn_npu"
    DIRECTML = "directml"
    CPU = "cpu"


@dataclass(frozen=True)
class BackendStatus:
    """Describes the selected inference backend and why it was chosen."""

    kind: BackendKind
    reason: str
    accelerated: bool = False


class InferenceBackend:
    """Small interface for enhancement inference backends."""

    status: BackendStatus

    @property
    def kind(self) -> BackendKind:
        return self.status.kind

    @property
    def accelerated(self) -> bool:
        return self.status.accelerated

    def analyze(self, frame: AudioFrame) -> FrameFeatures:
        return extract_features(frame)

    def decide(
        self,
        features: FrameFeatures,
        service: ServiceProfile,
        settings: EnhancementSettings,
    ) -> InferenceDecision:
        raise NotImplementedError


class HeuristicCpuBackend(InferenceBackend):
    """Deterministic fallback that emulates model decisions for development."""

    def __init__(self, reason: str = "ONNX Runtime QNN provider is unavailable") -> None:
        self.status = BackendStatus(BackendKind.CPU, reason, accelerated=False)

    def decide(
        self,
        features: FrameFeatures,
        service: ServiceProfile,
        settings: EnhancementSettings,
    ) -> InferenceDecision:
        compression = max(0.0, min(1.0, 1.0 - (features.crest_factor / 10.0)))
        quietness = max(0.0, min(1.0, (0.18 - features.rms) / 0.18))
        density = max(0.0, min(1.0, features.high_band_energy + features.low_band_energy))

        service_focus = {
            ServiceProfile.SPOTIFY: 1.10,
            ServiceProfile.APPLE_MUSIC: 0.85,
            ServiceProfile.YOUTUBE_MUSIC: 1.15,
            ServiceProfile.GENERIC: 1.0,
        }[service]
        clarity = settings.presence_tilt_db * service_focus * (0.25 + density * 0.35)
        bass = settings.bass_tilt_db * (0.35 + (1.0 - density) * 0.30)
        low_volume = quietness * min(3.0, settings.max_gain_db * 0.25)

        return InferenceDecision(
            bass_boost_db=bass,
            clarity_boost_db=clarity,
            low_volume_compensation_db=low_volume,
            transient_restore=compression * 0.35,
            backend_name=self.status.kind.value,
        )


class OnnxQnnBackend(HeuristicCpuBackend):
    """Placeholder for ONNX Runtime QNN execution provider integration.

    The public interface is intentionally the same as the CPU backend so the
    DSP pipeline can be tested without NPU hardware. Model loading and tensor
    binding should replace ``infer`` when a QNN-compatible model is added.
    """

    def __init__(self) -> None:
        super().__init__("ONNX Runtime QNN Execution Provider selected")
        self.status = BackendStatus(
            BackendKind.QNN_NPU,
            "ONNX Runtime QNN Execution Provider selected",
            accelerated=True,
        )


class DirectMlBackend(HeuristicCpuBackend):
    """Windows GPU/NPU-adjacent fallback used when QNN is not available."""

    def __init__(self) -> None:
        super().__init__("DirectML fallback selected")
        self.status = BackendStatus(BackendKind.DIRECTML, "DirectML fallback selected", True)


def extract_features(frame: AudioFrame) -> FrameFeatures:
    if not frame.samples:
        return FrameFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    rms = frame.rms
    peak = frame.peak
    crest_factor = peak / rms if rms else 0.0
    low_sum = 0.0
    high_sum = 0.0
    dot = 0.0
    left_energy = 0.0
    right_energy = 0.0
    previous_mid = 0.0

    for left, right in frame.samples:
        mid = (left + right) * 0.5
        low_sum += abs(mid)
        high_sum += abs(mid - previous_mid)
        dot += left * right
        left_energy += left * left
        right_energy += right * right
        previous_mid = mid

    denominator = (left_energy * right_energy) ** 0.5
    correlation = dot / denominator if denominator else 0.0
    count = len(frame.samples)
    return FrameFeatures(
        rms=rms,
        peak=peak,
        crest_factor=crest_factor,
        stereo_correlation=max(-1.0, min(1.0, correlation)),
        low_band_energy=min(1.0, low_sum / count),
        high_band_energy=min(1.0, high_sum / count),
    )


def _has_onnxruntime() -> bool:
    return importlib.util.find_spec("onnxruntime") is not None


def _is_arm64() -> bool:
    machine = platform.machine().lower()
    return machine in {"arm64", "aarch64"}


def select_backend(prefer_npu: bool = True, force_cpu: bool = False) -> InferenceBackend:
    """Select the best available inference backend for the current machine."""

    forced = os.getenv("SNAPDRAGON_AUDIO_BACKEND", "").strip().lower()
    if force_cpu or forced == BackendKind.CPU.value:
        return HeuristicCpuBackend("CPU backend forced")

    if prefer_npu and forced in {"", BackendKind.QNN_NPU.value}:
        if _is_arm64() and _has_onnxruntime():
            return OnnxQnnBackend()
        if forced == BackendKind.QNN_NPU.value:
            return HeuristicCpuBackend(
                "QNN NPU was forced but ARM64 ONNX Runtime support was not detected"
            )

    if forced == BackendKind.DIRECTML.value:
        return DirectMlBackend()

    return HeuristicCpuBackend("Using portable CPU fallback")
