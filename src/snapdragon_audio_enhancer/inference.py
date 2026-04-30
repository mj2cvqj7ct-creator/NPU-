"""NPU backend selection and deterministic inference shim.

The real Snapdragon X implementation should load an ONNX model through ONNX
Runtime's QNN Execution Provider or Qualcomm QNN. CI and non-Windows machines
cannot exercise that path, so this module exposes the same decision points while
falling back to a deterministic local estimator.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .dsp import SampleFrame, measure


class InferenceBackend(str, Enum):
    QNN_NPU = "qnn_npu"
    DIRECTML = "directml"
    CPU = "cpu"


@dataclass(frozen=True)
class InferenceDecision:
    backend: InferenceBackend
    reason: str


@dataclass(frozen=True)
class EnhancementHints:
    """Small set of controls a neural model would produce per audio block."""

    detail_gain_db: float
    density: float
    confidence: float


def select_backend() -> InferenceDecision:
    """Prefer Snapdragon NPU when the runtime advertises QNN availability."""

    forced = os.getenv("SNAPDRAGON_AUDIO_BACKEND", "").strip().lower()
    if forced:
        mapping = {
            "qnn": InferenceBackend.QNN_NPU,
            "qnn_npu": InferenceBackend.QNN_NPU,
            "directml": InferenceBackend.DIRECTML,
            "cpu": InferenceBackend.CPU,
        }
        if forced in mapping:
            return InferenceDecision(mapping[forced], "selected by SNAPDRAGON_AUDIO_BACKEND")

    machine = platform.machine().lower()
    has_qnn = os.getenv("QNN_SDK_ROOT") or os.getenv("ORT_QNN_AVAILABLE") == "1"
    if machine in {"arm64", "aarch64"} and has_qnn:
        return InferenceDecision(
            InferenceBackend.QNN_NPU,
            "ARM64 machine with QNN runtime markers available",
        )
    if platform.system() == "Windows":
        return InferenceDecision(
            InferenceBackend.DIRECTML,
            "Windows fallback when QNN is not configured",
        )
    return InferenceDecision(
        InferenceBackend.CPU,
        "portable fallback for development and CI",
    )


class NpuEnhancementModel:
    """Facade for block-level enhancement inference.

    The CPU path intentionally mirrors the interface of a future ONNX/QNN model:
    it takes normalized stereo PCM frames and returns bounded enhancement hints.
    """

    def __init__(self, decision: InferenceDecision | None = None) -> None:
        self.decision = decision or select_backend()

    def infer(self, frames: Sequence[SampleFrame]) -> EnhancementHints:
        stats = measure(frames)
        density = max(0.0, min(1.0, (stats.rms_dbfs + 36.0) / 24.0))
        transient_room = max(0.0, min(1.0, (stats.crest_factor_db - 7.0) / 10.0))

        # Leave already-dense masters mostly alone and add clarity when there is
        # transient headroom. A trained NPU model can replace this heuristic.
        detail_gain_db = min(1.2, transient_room * (1.0 - density) * 2.0)
        confidence = 0.55 if self.decision.backend is InferenceBackend.CPU else 0.8
        return EnhancementHints(
            detail_gain_db=detail_gain_db,
            density=density,
            confidence=confidence,
        )
