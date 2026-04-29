"""Inference backends for Snapdragon X NPU-assisted enhancement controls."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

import numpy as np

from .dsp import AudioFeatures, EnhancementControls


class InferenceProvider(str, Enum):
    """Supported inference provider categories."""

    QNN = "qnn"
    DIRECTML = "directml"
    CPU = "cpu"


class InferenceBackend(Protocol):
    provider: InferenceProvider

    def infer(self, features: AudioFeatures) -> EnhancementControls:
        """Infer enhancement controls from short-window audio features."""


class HeuristicInferenceBackend:
    """CPU fallback that mirrors the model contract without requiring an NPU."""

    provider = InferenceProvider.CPU

    def infer(self, features: AudioFeatures) -> EnhancementControls:
        crowded_vocal_band = float(np.clip(features.vocal_band_energy * 8.0, 0.0, 1.0))
        dark_mix = float(np.clip(1.0 - features.high_band_energy * 20.0, 0.0, 1.0))
        bass_sparse = float(np.clip(1.0 - features.low_band_energy * 14.0, 0.0, 1.0))
        clipping_guard = -1.5 if features.clipping_ratio > 0.001 else 0.0
        return EnhancementControls(
            pre_gain_db=clipping_guard,
            bass_gain_db=0.4 + 1.4 * bass_sparse,
            presence_gain_db=0.3 + 1.3 * (1.0 - crowded_vocal_band),
            air_gain_db=0.2 + 1.5 * dark_mix,
            stereo_width=1.04 if features.stereo_correlation > 0.1 else 1.0,
            compressor_threshold_db=-18.0,
            compressor_ratio=1.7,
            limiter_ceiling_db=-1.0,
        )


class OnnxRuntimeInferenceBackend:
    """ONNX Runtime backend with QNN, DirectML, or CPU provider selection."""

    def __init__(
        self,
        model_path: Path,
        provider: InferenceProvider = InferenceProvider.QNN,
    ) -> None:
        self.provider = provider
        self.model_path = Path(model_path)
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError("Install the 'onnx' extra to enable ONNX Runtime inference.") from exc

        provider_names = self._provider_names(provider)
        self._session = ort.InferenceSession(str(self.model_path), providers=provider_names)
        self._input_name = self._session.get_inputs()[0].name

    def infer(self, features: AudioFeatures) -> EnhancementControls:
        vector = np.array(
            [
                features.rms_db,
                features.peak_db,
                features.crest_factor_db,
                features.spectral_centroid_hz,
                features.low_band_energy,
                features.vocal_band_energy,
                features.high_band_energy,
                features.clipping_ratio,
                features.stereo_correlation,
                features.transient_density,
            ],
            dtype=np.float32,
        )[np.newaxis, :]
        output = np.asarray(self._session.run(None, {self._input_name: vector})[0], dtype=np.float32)
        controls = output.reshape(-1)
        if controls.size < 9:
            raise ValueError("Enhancement model must output nine DSP controls.")
        return EnhancementControls(
            pre_gain_db=float(np.clip(controls[0], -6.0, 6.0)),
            bass_gain_db=float(np.clip(controls[1], -3.0, 3.0)),
            presence_gain_db=float(np.clip(controls[2], -3.0, 3.0)),
            air_gain_db=float(np.clip(controls[3], -3.0, 3.0)),
            stereo_width=float(np.clip(controls[4], 0.5, 1.4)),
            transient_gain_db=float(np.clip(controls[5], -1.5, 2.0)),
            compressor_threshold_db=float(np.clip(controls[6], -36.0, -6.0)),
            compressor_ratio=float(np.clip(controls[7], 1.0, 6.0)),
            limiter_ceiling_db=float(np.clip(controls[8], -6.0, -0.1)),
        )

    @staticmethod
    def _provider_names(provider: InferenceProvider) -> list[str]:
        if provider == InferenceProvider.QNN:
            return ["QNNExecutionProvider", "CPUExecutionProvider"]
        if provider == InferenceProvider.DIRECTML:
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]


def build_backend(
    model_path: str | Path | None = None,
    provider: InferenceProvider = InferenceProvider.QNN,
) -> InferenceBackend:
    """Build the preferred backend, falling back to deterministic heuristics."""

    if model_path is None:
        return HeuristicInferenceBackend()
    try:
        return OnnxRuntimeInferenceBackend(Path(model_path), provider=provider)
    except Exception:
        return HeuristicInferenceBackend()
