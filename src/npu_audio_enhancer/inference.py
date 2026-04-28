"""NPU inference backend selection scaffolding.

The real Snapdragon X path should run through ONNX Runtime with the QNN
Execution Provider or a Qualcomm AI Engine Direct integration. This module keeps
that decision isolated so the DSP pipeline can run safely with a deterministic
CPU fallback when the target hardware or runtime is absent.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence


class InferenceBackend(str, Enum):
    """Supported inference execution targets in preference order."""

    QNN = "qnn"
    DIRECTML = "directml"
    CPU = "cpu"


@dataclass(frozen=True)
class InferenceRequest:
    """Small feature vector passed to the enhancement model."""

    spectral_centroid_hz: float
    low_band_energy: float
    mid_band_energy: float
    high_band_energy: float
    loudness_dbfs: float
    service_hint: str = "generic"

    def as_vector(self) -> tuple[float, ...]:
        return (
            self.spectral_centroid_hz,
            self.low_band_energy,
            self.mid_band_energy,
            self.high_band_energy,
            self.loudness_dbfs,
        )


def select_backend(
    *,
    preferred: Sequence[InferenceBackend | str] | None = None,
    environment: Mapping[str, str] | None = None,
) -> InferenceBackend:
    """Select the best available inference backend.

    Environment overrides are intentionally simple so tests and field diagnostics
    can force behavior without probing platform-specific SDKs:

    - ``NPU_AUDIO_FORCE_BACKEND=qnn|directml|cpu``
    - ``NPU_AUDIO_DISABLE_QNN=1``
    - ``NPU_AUDIO_DISABLE_DIRECTML=1``
    """

    env = os.environ if environment is None else environment
    forced = env.get("NPU_AUDIO_FORCE_BACKEND")
    if forced:
        return InferenceBackend(forced.lower())

    order = tuple(_coerce_backend(item) for item in preferred) if preferred else (
        InferenceBackend.QNN,
        InferenceBackend.DIRECTML,
        InferenceBackend.CPU,
    )

    for backend in order:
        if _is_backend_available(backend, env):
            return backend

    return InferenceBackend.CPU


def run_personalization_inference(
    request: InferenceRequest,
    *,
    backend: InferenceBackend | str | None = None,
) -> dict[str, float]:
    """Return bounded enhancement controls for a frame.

    This deterministic implementation mirrors the contract expected from a tiny
    neural model: estimate conservative EQ and stereo adjustments from extracted
    features. A future QNN/ONNX implementation should preserve the same keys and
    value ranges.
    """

    selected = _coerce_backend(backend) if backend is not None else select_backend()
    vector = request.as_vector()
    low, mid, high = request.low_band_energy, request.mid_band_energy, request.high_band_energy
    total = max(low + mid + high, 1.0e-9)

    low_ratio = low / total
    high_ratio = high / total
    mid_ratio = mid / total

    clarity = _clamp((mid_ratio - 0.32) * 2.0, -0.18, 0.22)
    warmth = _clamp((0.38 - low_ratio) * 1.3, -0.16, 0.18)
    air = _clamp((0.24 - high_ratio) * 1.0, -0.12, 0.16)

    if request.loudness_dbfs < -28.0:
        warmth += 0.04
        clarity += 0.03

    # Keep fallback outputs intentionally conservative. The backend is surfaced
    # for telemetry without changing the safety envelope.
    backend_bias = 0.0 if selected is InferenceBackend.CPU else 0.01
    return {
        "low_shelf_gain": _clamp(warmth + backend_bias, -0.18, 0.20),
        "presence_gain": _clamp(clarity + backend_bias, -0.18, 0.24),
        "air_gain": _clamp(air, -0.12, 0.16),
        "stereo_width": _clamp(1.0 + (request.spectral_centroid_hz - 1800.0) / 20000.0, 0.92, 1.08),
    }


def _coerce_backend(value: InferenceBackend | str) -> InferenceBackend:
    if isinstance(value, InferenceBackend):
        return value
    return InferenceBackend(value.lower())


def _is_backend_available(backend: InferenceBackend, env: Mapping[str, str]) -> bool:
    if backend is InferenceBackend.CPU:
        return True
    if backend is InferenceBackend.QNN:
        if env.get("NPU_AUDIO_DISABLE_QNN") == "1":
            return False
        return _onnxruntime_has_provider("QNNExecutionProvider")
    if backend is InferenceBackend.DIRECTML:
        if env.get("NPU_AUDIO_DISABLE_DIRECTML") == "1":
            return False
        return _onnxruntime_has_provider("DmlExecutionProvider")
    return False


def _onnxruntime_has_provider(provider: str) -> bool:
    if importlib.util.find_spec("onnxruntime") is None:
        return False
    try:
        import onnxruntime as ort  # type: ignore[import-not-found]
    except Exception:
        return False
    return provider in set(ort.get_available_providers())


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
