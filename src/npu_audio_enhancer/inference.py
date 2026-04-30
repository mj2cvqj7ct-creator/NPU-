"""Inference provider selection for Snapdragon X NPU assisted enhancement."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from .audio_frame import AudioFrame


class ProviderKind(str, Enum):
    """Supported inference backends in preference order."""

    QNN = "qnn"
    DIRECTML = "directml"
    CPU = "cpu"


@dataclass(frozen=True)
class EnhancementHints:
    """Low-bandwidth controls predicted by the enhancement model."""

    clarity_db: float = 0.0
    bass_db: float = 0.0
    stereo_width: float = 1.0
    limiter_ceiling_dbfs: float = -1.0


class InferenceProvider(Protocol):
    """Predict enhancement controls for a short PCM frame."""

    kind: ProviderKind

    def predict(self, frame: AudioFrame) -> EnhancementHints:
        """Return model-derived enhancement controls."""


@dataclass
class CpuFallbackProvider:
    """Deterministic fallback used until a trained ONNX model is available."""

    kind: ProviderKind = ProviderKind.CPU

    def predict(self, frame: AudioFrame) -> EnhancementHints:
        rms = frame.rms()
        peak = frame.peak()
        low_level = rms < 0.08
        crowded = peak > 0.85 and rms > 0.22
        return EnhancementHints(
            clarity_db=1.5 if low_level else 0.8,
            bass_db=0.8 if low_level else 0.2,
            stereo_width=0.96 if crowded else 1.04,
            limiter_ceiling_dbfs=-1.2,
        )


@dataclass
class OnnxRuntimeProvider:
    """ONNX Runtime provider placeholder for QNN or DirectML execution."""

    kind: ProviderKind
    model_path: str
    session: object | None = None

    def __post_init__(self) -> None:
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional env
            raise RuntimeError("onnxruntime is not installed") from exc

        providers = {
            ProviderKind.QNN: ["QNNExecutionProvider", "CPUExecutionProvider"],
            ProviderKind.DIRECTML: ["DmlExecutionProvider", "CPUExecutionProvider"],
        }[self.kind]
        self.session = ort.InferenceSession(self.model_path, providers=providers)

    def predict(self, frame: AudioFrame) -> EnhancementHints:
        if self.session is None:
            raise RuntimeError("ONNX Runtime session is not initialized")

        try:
            import numpy as np  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on optional env
            raise RuntimeError("numpy is required for ONNX Runtime inference") from exc

        input_name = self.session.get_inputs()[0].name
        pcm = np.asarray(frame.samples, dtype=np.float32)[None, :, :]
        outputs = self.session.run(None, {input_name: pcm})
        values = outputs[0][0]
        return EnhancementHints(
            clarity_db=float(values[0]),
            bass_db=float(values[1]),
            stereo_width=float(values[2]),
            limiter_ceiling_dbfs=float(values[3]),
        )


def select_provider(model_path: str | None = None) -> InferenceProvider:
    """Select QNN on Snapdragon X when configured, then DirectML, then CPU."""

    requested = os.getenv("NPU_AUDIO_PROVIDER", "auto").lower()
    candidates: list[ProviderKind]
    if requested == "auto":
        candidates = [ProviderKind.QNN, ProviderKind.DIRECTML, ProviderKind.CPU]
    else:
        try:
            candidates = [ProviderKind(requested)]
        except ValueError as exc:
            raise ValueError(f"Unsupported NPU_AUDIO_PROVIDER: {requested}") from exc

    for kind in candidates:
        if kind == ProviderKind.CPU:
            return CpuFallbackProvider()
        if model_path is None:
            continue
        try:
            return OnnxRuntimeProvider(kind=kind, model_path=model_path)
        except RuntimeError:
            if requested != "auto":
                raise
    return CpuFallbackProvider()
