"""NPU backend selection and lightweight inference abstraction.

The production target is ONNX Runtime with Qualcomm's QNN Execution Provider on
Snapdragon X.  This module keeps the runtime-facing surface small so the DSP
pipeline can be exercised on development machines without Qualcomm hardware.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from math import sqrt
from typing import Mapping

from .audio import AudioBuffer
from .dsp import clamp, linear_to_db


class BackendKind(str, Enum):
    """Inference backend preference in descending hardware specificity."""

    QNN = "qnn"
    DIRECTML = "directml"
    CPU = "cpu"


@dataclass(frozen=True)
class AudioFeatures:
    """Compact frame features suitable for local preference and quality models."""

    rms_dbfs: float
    low_band_energy: float
    mid_band_energy: float
    high_band_energy: float
    stereo_width: float


@dataclass(frozen=True)
class EnhancementHints:
    """Model outputs consumed by the deterministic DSP stage."""

    clarity: float
    warmth: float
    width: float


def select_backend(
    preferred: BackendKind | str | None = None,
    env: Mapping[str, str] | None = None,
    available_providers: tuple[str, ...] | None = None,
) -> BackendKind:
    """Choose the best available inference backend.

    Environment override:
    - ``SNAPDRAGON_AUDIO_BACKEND=qnn|directml|cpu``

    Provider names match ONNX Runtime conventions when the package is present.
    """

    if preferred is not None:
        if isinstance(preferred, BackendKind):
            return preferred
        normalized_preferred = preferred.strip().lower()
        for backend in BackendKind:
            if normalized_preferred == backend.value:
                return backend
        raise ValueError(f"Unsupported preferred backend: {preferred!r}")

    source = env if env is not None else os.environ
    override = source.get("SNAPDRAGON_AUDIO_BACKEND")
    if override:
        normalized = override.strip().lower()
        for backend in BackendKind:
            if normalized == backend.value:
                return backend
        raise ValueError(f"Unsupported SNAPDRAGON_AUDIO_BACKEND: {override!r}")

    providers = set(available_providers or _detect_onnx_providers())
    if "QNNExecutionProvider" in providers:
        return BackendKind.QNN
    if "DmlExecutionProvider" in providers:
        return BackendKind.DIRECTML
    return BackendKind.CPU


def _detect_onnx_providers() -> tuple[str, ...]:
    try:
        import onnxruntime as ort  # type: ignore
    except Exception:
        return ()
    return tuple(ort.get_available_providers())


class NpuAssistModel:
    """Small model facade used by the real-time enhancement pipeline.

    If no ONNX model is supplied, a deterministic CPU heuristic is used.  The
    heuristic mirrors the output contract expected from an eventual QNN model,
    allowing the rest of the pipeline and tests to remain stable.
    """

    def __init__(
        self,
        backend: BackendKind | None = None,
        model_path: str | None = None,
        providers: tuple[str, ...] | None = None,
    ) -> None:
        self.backend = backend or select_backend(available_providers=providers)
        self.model_path = model_path
        self._session = None
        if model_path is not None:
            self._session = self._create_session(model_path)

    def infer(self, buffer: AudioBuffer) -> EnhancementHints:
        features = extract_features(buffer)
        if self._session is not None:
            return self._infer_onnx(features)
        return heuristic_hints(features)

    def predict_controls(self, metrics: object) -> dict[str, float]:
        """Return DSP controls from metrics until a real model is configured."""

        peak_dbfs = float(getattr(metrics, "peak_dbfs"))
        rms_dbfs = float(getattr(metrics, "rms_dbfs"))
        crest_db = peak_dbfs - rms_dbfs
        quiet = clamp((-18.0 - rms_dbfs) / 24.0, 0.0, 1.0)
        compressed = clamp((12.0 - crest_db) / 8.0, 0.0, 1.0)
        return {
            "low_shelf_db": 0.7 * quiet,
            "clarity_db": 0.8 + 0.7 * compressed,
            "air_db": 0.4 + 0.4 * compressed,
            "stereo_width_delta": 0.02 * quiet,
        }

    def _create_session(self, model_path: str):
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:
            raise RuntimeError("onnxruntime is required when model_path is set") from exc

        provider_order = {
            BackendKind.QNN: ["QNNExecutionProvider", "CPUExecutionProvider"],
            BackendKind.DIRECTML: ["DmlExecutionProvider", "CPUExecutionProvider"],
            BackendKind.CPU: ["CPUExecutionProvider"],
        }[self.backend]
        return ort.InferenceSession(model_path, providers=provider_order)

    def _infer_onnx(self, features: AudioFeatures) -> EnhancementHints:
        assert self._session is not None
        import numpy as np

        input_name = self._session.get_inputs()[0].name
        output = self._session.run(None, {input_name: np.array([features_tuple(features)], dtype=np.float32)})[0]
        clarity, warmth, width = np.asarray(output, dtype=np.float32).reshape(-1)[:3]
        return EnhancementHints(
            clarity=float(np.clip(clarity, 0.0, 1.0)),
            warmth=float(np.clip(warmth, 0.0, 1.0)),
            width=float(np.clip(width, 0.0, 1.0)),
        )


def extract_features(buffer: AudioBuffer) -> AudioFeatures:
    mono = buffer.mono()
    if not mono:
        return AudioFeatures(-120.0, 0.0, 0.0, 0.0, 0.0)

    low_energy = 0.0
    high_energy = 0.0
    total = 0.0
    previous = mono[0]
    for sample in mono:
        low = 0.98 * previous + 0.02 * sample
        high = sample - previous
        previous = low
        low_energy += low * low
        high_energy += high * high
        total += sample * sample
    total += 1.0e-12
    low_ratio = clamp(low_energy / total, 0.0, 1.0)
    high_ratio = clamp(high_energy / total, 0.0, 1.0)
    mid_ratio = clamp(1.0 - low_ratio - high_ratio, 0.0, 1.0)

    if buffer.channels >= 2:
        side_sum = 0.0
        mid_sum = 0.0
        for left, right in buffer.frames:
            side = left - right
            mid = left + right
            side_sum += side * side
            mid_sum += mid * mid
        width = sqrt(side_sum / (mid_sum + 1.0e-12))
    else:
        width = 0.0

    return AudioFeatures(
        rms_dbfs=linear_to_db(buffer.rms()),
        low_band_energy=low_ratio,
        mid_band_energy=mid_ratio,
        high_band_energy=high_ratio,
        stereo_width=clamp(width, 0.0, 1.0),
    )


def heuristic_hints(features: AudioFeatures) -> EnhancementHints:
    """Approximate model behavior for development and CPU fallback."""

    dullness = clamp(0.35 - features.high_band_energy, 0.0, 0.35) / 0.35
    thinness = clamp(0.20 - features.low_band_energy, 0.0, 0.20) / 0.20
    narrowness = 1.0 - clamp(features.stereo_width, 0.0, 1.0)
    quiet = clamp((-18.0 - features.rms_dbfs) / 24.0, 0.0, 1.0)

    return EnhancementHints(
        clarity=clamp(0.25 + 0.50 * dullness + 0.20 * quiet, 0.0, 1.0),
        warmth=clamp(0.20 + 0.45 * thinness + 0.10 * quiet, 0.0, 1.0),
        width=clamp(0.15 + 0.35 * narrowness, 0.0, 0.65),
    )


def features_tuple(features: AudioFeatures) -> tuple[float, float, float, float, float]:
    return (
        features.rms_dbfs,
        features.low_band_energy,
        features.mid_band_energy,
        features.high_band_energy,
        features.stereo_width,
    )
