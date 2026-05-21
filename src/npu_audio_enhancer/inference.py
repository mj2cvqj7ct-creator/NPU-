"""NPU backend discovery for Snapdragon X audio enhancement."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from typing import Protocol

from .dsp import FrameFeatures


class BackendChoice(str, Enum):
    """Execution backends ordered by preference for ARM64 Snapdragon X."""

    QNN_NPU = "qnn-npu"
    DIRECTML = "directml"
    CPU = "cpu"


@dataclass(frozen=True)
class BackendStatus:
    choice: BackendChoice
    provider: str
    reason: str


class FeatureModel(Protocol):
    """Small model interface for local music preference and tone inference."""

    def infer(self, features: FrameFeatures) -> dict[str, float]:
        """Return bounded DSP control values such as EQ gain and width."""


class InferenceBackendSelector:
    """Selects ONNX Runtime providers without making hard runtime dependencies."""

    _QNN_PROVIDER = "QNNExecutionProvider"
    _DIRECTML_PROVIDER = "DmlExecutionProvider"
    _CPU_PROVIDER = "CPUExecutionProvider"

    def __init__(self, available_providers: list[str] | None = None) -> None:
        self._available_providers = available_providers

    def select(self) -> BackendStatus:
        providers = self._available_providers
        if providers is None:
            providers = self._load_onnxruntime_providers()

        if self._QNN_PROVIDER in providers and self._looks_like_snapdragon_arm64():
            return BackendStatus(
                BackendChoice.QNN_NPU,
                self._QNN_PROVIDER,
                "Snapdragon ARM64 host with ONNX Runtime QNN provider available",
            )
        if self._DIRECTML_PROVIDER in providers:
            return BackendStatus(
                BackendChoice.DIRECTML,
                self._DIRECTML_PROVIDER,
                "QNN NPU provider unavailable; using Windows DirectML fallback",
            )
        if self._CPU_PROVIDER in providers:
            return BackendStatus(
                BackendChoice.CPU,
                self._CPU_PROVIDER,
                "QNN and DirectML unavailable; using CPU fallback",
            )
        return BackendStatus(
            BackendChoice.CPU,
            self._CPU_PROVIDER,
            "onnxruntime is not installed; using built-in rule-based DSP only",
        )

    def _load_onnxruntime_providers(self) -> list[str]:
        try:
            onnxruntime = import_module("onnxruntime")
        except ModuleNotFoundError:
            return []
        return list(onnxruntime.get_available_providers())

    def _looks_like_snapdragon_arm64(self) -> bool:
        machine = platform.machine().lower()
        processor = platform.processor().lower()
        platform_name = platform.platform().lower()
        is_arm64 = machine in {"arm64", "aarch64"} or "arm" in processor
        is_snapdragon = any(
            token in f"{processor} {platform_name}" for token in ("snapdragon", "qualcomm", "oryon")
        )
        return is_arm64 and is_snapdragon


class HeuristicFeatureModel:
    """Tiny local model used until a quantized ONNX model is available."""

    def infer(self, features: FrameFeatures) -> dict[str, float]:
        bass = 0.0
        if features.low_band_energy < 0.25:
            bass = 1.0
        elif features.low_band_energy > 0.55:
            bass = -0.75

        presence = 0.75 if features.mid_band_energy < 0.2 else 0.0
        if features.crest_factor_db < 8.0:
            presence -= 0.5

        width = 1.02
        if features.stereo_correlation > 0.9:
            width = 1.06
        elif features.stereo_correlation < 0.2:
            width = 1.0

        return {
            "bass_gain_db": _bound(bass, -2.0, 2.0),
            "presence_gain_db": _bound(presence, -1.5, 1.5),
            "stereo_width": _bound(width, 1.0, 1.08),
        }


def _bound(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
