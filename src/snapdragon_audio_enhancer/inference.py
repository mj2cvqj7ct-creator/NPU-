"""Inference provider abstraction for Snapdragon X NPU assisted enhancement."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math
import os
import platform
from typing import Protocol

from .audio_types import AudioFrame


class InferenceProvider(Protocol):
    """Small interface used by the DSP pipeline for AI-assisted controls."""

    name: str

    def analyze(self, frame: AudioFrame) -> "EnhancementControls":
        """Return adaptive enhancement controls for a short audio frame."""


@dataclass(frozen=True)
class EnhancementControls:
    """Per-frame controls normally produced by an ML model."""

    clarity: float
    bass_tightness: float
    stereo_width: float
    transient_restore: float

    def clamp(self) -> "EnhancementControls":
        return EnhancementControls(
            clarity=_clamp(self.clarity, 0.0, 1.0),
            bass_tightness=_clamp(self.bass_tightness, 0.0, 1.0),
            stereo_width=_clamp(self.stereo_width, 0.0, 1.0),
            transient_restore=_clamp(self.transient_restore, 0.0, 1.0),
        )


@dataclass(frozen=True)
class ProviderSelection:
    """Selected inference backend and the reason it was chosen."""

    provider: InferenceProvider
    reason: str


class HeuristicInferenceProvider:
    """Deterministic CPU fallback that approximates model controls."""

    name = "heuristic-cpu"

    def analyze(self, frame: AudioFrame) -> EnhancementControls:
        rms = frame.rms()
        peak = frame.peak()
        crest = peak / max(rms, 1.0e-6)
        side_ratio = frame.mid_side_ratio()

        clarity = _clamp(0.25 + 2.2 * rms - 0.03 * crest, 0.0, 0.85)
        bass_tightness = _clamp(0.55 - 1.3 * rms + 0.015 * crest, 0.0, 0.9)
        stereo_width = _clamp(0.35 + 0.9 * side_ratio, 0.0, 0.75)
        transient_restore = _clamp((crest - 4.0) / 10.0, 0.0, 0.7)
        return EnhancementControls(
            clarity=clarity,
            bass_tightness=bass_tightness,
            stereo_width=stereo_width,
            transient_restore=transient_restore,
        )


class OnnxQnnInferenceProvider:
    """Thin placeholder for ONNX Runtime QNN EP integration."""

    name = "onnxruntime-qnn"

    def __init__(self) -> None:
        if importlib.util.find_spec("onnxruntime") is None:
            raise RuntimeError("onnxruntime is not installed")

    def analyze(self, frame: AudioFrame) -> EnhancementControls:
        # A production build loads a compact ONNX model with QNN EP here.
        # Keep the current path deterministic until model artifacts are added.
        return HeuristicInferenceProvider().analyze(frame)


def select_provider(prefer_npu: bool = True) -> ProviderSelection:
    """Choose the best available local inference provider.

    Snapdragon X machines running Windows on ARM can use ONNX Runtime's QNN
    Execution Provider once the dependency and model are present. Other hosts
    intentionally fall back to a deterministic CPU heuristic so development and
    tests remain portable.
    """

    force_cpu = os.getenv("SNAPDRAGON_AUDIO_FORCE_CPU", "").strip() == "1"
    machine = platform.machine().lower()
    system = platform.system().lower()
    likely_snapdragon_x = machine in {"arm64", "aarch64"} and system == "windows"

    if prefer_npu and not force_cpu and likely_snapdragon_x:
        try:
            return ProviderSelection(
                provider=OnnxQnnInferenceProvider(),
                reason="Windows ARM64 host detected; ONNX Runtime QNN path is available.",
            )
        except RuntimeError as exc:
            return ProviderSelection(
                provider=HeuristicInferenceProvider(),
                reason=f"QNN path unavailable ({exc}); using CPU fallback.",
            )

    if force_cpu:
        reason = "SNAPDRAGON_AUDIO_FORCE_CPU=1 set; using CPU fallback."
    elif not prefer_npu:
        reason = "NPU preference disabled; using CPU fallback."
    else:
        reason = "Non-Windows ARM64 host; using portable CPU fallback."
    return ProviderSelection(provider=HeuristicInferenceProvider(), reason=reason)


def _clamp(value: float, low: float, high: float) -> float:
    if math.isnan(value):
        return low
    return min(high, max(low, value))
