"""NPU-aware inference provider selection.

The production path for Snapdragon X is ONNX Runtime with the QNN Execution
Provider. This module keeps that dependency optional so DSP development and
tests remain runnable on generic Linux CI hosts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class EnhancementBackend(Protocol):
    """Backend interface for frame-wise enhancement models."""

    name: str

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """Return an enhanced copy of a stereo float frame."""


@dataclass(frozen=True)
class IdentityBackend:
    """Safe fallback when NPU inference is unavailable."""

    name: str = "identity-cpu"

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        return np.asarray(frame, dtype=np.float32).copy()


class OnnxRuntimeBackend:
    """ONNX Runtime wrapper with Snapdragon X QNN preference."""

    def __init__(self, model_path: str | Path, prefer_qnn: bool = True) -> None:
        try:
            import onnxruntime as ort  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "onnxruntime is not installed; install the onnx optional extra"
            ) from exc

        providers = ort.get_available_providers()
        requested: list[str] = []
        if prefer_qnn and "QNNExecutionProvider" in providers:
            requested.append("QNNExecutionProvider")
        if "DmlExecutionProvider" in providers:
            requested.append("DmlExecutionProvider")
        requested.append("CPUExecutionProvider")

        self._session = ort.InferenceSession(
            str(model_path),
            providers=[provider for provider in requested if provider in providers],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self.name = "+".join(self._session.get_providers())

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        batch = np.asarray(frame, dtype=np.float32)[None, ...]
        output = self._session.run([self._output_name], {self._input_name: batch})[0]
        enhanced = np.asarray(output[0], dtype=np.float32)
        if enhanced.shape != frame.shape:
            raise ValueError(
                f"model returned {enhanced.shape}, expected frame shape {frame.shape}"
            )
        return enhanced


def create_backend(model_path: str | Path | None = None) -> EnhancementBackend:
    """Create the best available backend for the current machine.

    If a model path is supplied, ONNX Runtime tries QNN first, then DirectML,
    then CPU. Without a model, an identity backend is used so the deterministic
    DSP stages can still run.
    """

    if model_path is None:
        return IdentityBackend()
    return OnnxRuntimeBackend(model_path)
