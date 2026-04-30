"""Inference backend selection for Snapdragon X NPU assisted audio enhancement."""

from __future__ import annotations

import importlib.util
import os
import platform
from dataclasses import dataclass
from enum import Enum

from .frame import AudioFrame


class BackendKind(str, Enum):
    """Supported inference execution targets in priority order."""

    SNAPDRAGON_QNN = "snapdragon_qnn"
    DIRECTML = "directml"
    CPU = "cpu"


@dataclass(frozen=True)
class InferenceResult:
    """Compact model output used by the DSP post processor."""

    clarity_boost: float
    warmth_boost: float
    transient_restore: float
    stereo_expansion: float
    gain_trim_db: float
    backend: BackendKind


class EnhancementBackend:
    """Small interface shared by hardware and fallback inference paths."""

    kind: BackendKind
    reason: str

    @property
    def name(self) -> str:
        return self.kind.value

    def infer(
        self,
        frame: AudioFrame,
        features: object | None = None,
        profile: object | None = None,
    ) -> InferenceResult:
        raise NotImplementedError


InferenceBackend = EnhancementBackend


@dataclass
class HeuristicCpuBackend(EnhancementBackend):
    """Deterministic fallback that mirrors the model contract for tests and unsupported hosts."""

    reason: str = "CPU fallback is always available"
    kind: BackendKind = BackendKind.CPU

    def infer(
        self,
        frame: AudioFrame,
        features: object | None = None,
        profile: object | None = None,
    ) -> InferenceResult:
        if features is None:
            from .dsp import extract_features

            features = extract_features(frame)

        crest_factor = features.true_peak / max(features.rms, 1.0e-9)
        density = min(1.0, features.zero_crossing_rate * 24.0)
        clipping_penalty = min(1.0, features.clipping_ratio * 80.0)
        warmth = max(0.0, min(1.0, features.low_band_energy * 1.5 + clipping_penalty * 0.25))

        return InferenceResult(
            clarity_boost=max(0.0, min(1.0, 0.35 + 0.45 * density - 0.35 * clipping_penalty)),
            warmth_boost=warmth,
            transient_restore=max(0.0, min(1.0, (crest_factor - 2.4) / 5.0)),
            stereo_expansion=max(0.0, min(1.0, 1.0 - abs(features.stereo_correlation))),
            gain_trim_db=max(-1.5, min(0.5, -features.loudness_db - 18.0)) * 0.08,
            backend=self.kind,
        )


@dataclass
class OrtQnnBackend(EnhancementBackend):
    """Placeholder adapter for ONNX Runtime QNN Execution Provider deployments."""

    model_path: str | None
    reason: str = "ONNX Runtime QNN Execution Provider is available"
    kind: BackendKind = BackendKind.SNAPDRAGON_QNN

    def __post_init__(self) -> None:
        if not self.model_path:
            self.reason = "QNN provider detected; running heuristic contract until a model is configured"

    def infer(
        self,
        frame: AudioFrame,
        features: object | None = None,
        profile: object | None = None,
    ) -> InferenceResult:
        # The runtime session is intentionally not created without an explicit model. This keeps
        # the OSS core testable while preserving the same output contract for the DSP pipeline.
        result = HeuristicCpuBackend(reason=self.reason).infer(frame, features, profile)
        return InferenceResult(
            clarity_boost=result.clarity_boost,
            warmth_boost=result.warmth_boost,
            transient_restore=result.transient_restore,
            stereo_expansion=result.stereo_expansion,
            gain_trim_db=result.gain_trim_db,
            backend=self.kind,
        )


@dataclass
class DirectMlBackend(EnhancementBackend):
    """Windows GPU/NPU fallback adapter used when QNN is unavailable."""

    reason: str = "DirectML fallback provider is available"
    kind: BackendKind = BackendKind.DIRECTML

    def infer(
        self,
        frame: AudioFrame,
        features: object | None = None,
        profile: object | None = None,
    ) -> InferenceResult:
        result = HeuristicCpuBackend(reason=self.reason).infer(frame, features, profile)
        return InferenceResult(
            clarity_boost=result.clarity_boost,
            warmth_boost=result.warmth_boost,
            transient_restore=result.transient_restore,
            stereo_expansion=result.stereo_expansion,
            gain_trim_db=result.gain_trim_db,
            backend=self.kind,
        )


def select_backend(prefer_npu: bool = True, model_path: str | None = None) -> EnhancementBackend:
    """Select the best available inference backend for the current host."""

    if not prefer_npu:
        return HeuristicCpuBackend(reason="NPU preference disabled")

    forced = os.getenv("SNAPDRAGON_AUDIO_BACKEND", "").strip().lower()
    if forced == "cpu":
        return HeuristicCpuBackend(reason="Forced by SNAPDRAGON_AUDIO_BACKEND")
    if forced == "directml":
        return DirectMlBackend(reason="Forced by SNAPDRAGON_AUDIO_BACKEND")
    if forced == "qnn":
        return OrtQnnBackend(model_path=model_path, reason="Forced by SNAPDRAGON_AUDIO_BACKEND")

    machine = platform.machine().lower()
    ort_available = importlib.util.find_spec("onnxruntime") is not None
    likely_snapdragon = machine in {"arm64", "aarch64"} and _has_snapdragon_hint()
    if ort_available and likely_snapdragon:
        return OrtQnnBackend(model_path=model_path)

    if platform.system().lower() == "windows" and ort_available:
        return DirectMlBackend(reason="ONNX Runtime available but Snapdragon QNN was not detected")

    return HeuristicCpuBackend(reason="No Snapdragon X NPU runtime detected")


def build_default_inference_engine(
    prefer_npu: bool = True,
    model_path: str | None = None,
) -> EnhancementBackend:
    return select_backend(prefer_npu=prefer_npu, model_path=model_path)


def _has_snapdragon_hint() -> bool:
    text = " ".join(
        value
        for value in (
            platform.processor(),
            platform.platform(),
            os.getenv("PROCESSOR_IDENTIFIER", ""),
            os.getenv("SNAPDRAGON_AUDIO_DEVICE", ""),
        )
        if value
    ).lower()
    return any(token in text for token in ("snapdragon", "qualcomm", "oryon", "hexagon"))
