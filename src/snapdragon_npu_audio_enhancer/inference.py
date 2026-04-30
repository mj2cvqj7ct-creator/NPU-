"""Inference backend selection for Snapdragon X NPU assisted enhancement."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

from .audio_frame import AudioFrame


class BackendKind(str, Enum):
    QNN = "qnn"
    DIRECTML = "directml"
    CPU = "cpu"


@dataclass(frozen=True)
class InferenceResult:
    """Control signals produced by the neural stage."""

    clarity_gain_db: float
    warmth_gain_db: float
    transient_boost_db: float
    stereo_width: float
    backend: BackendKind


class InferenceBackend:
    """Abstract inference backend contract."""

    kind: BackendKind

    def infer(self, frame: AudioFrame) -> InferenceResult:
        raise NotImplementedError


class HeuristicCpuBackend(InferenceBackend):
    """Deterministic fallback that mirrors a small model's control output.

    This keeps the pipeline useful and testable on non-Snapdragon development
    machines while reserving the same control contract for QNN/ONNX Runtime.
    """

    kind = BackendKind.CPU

    def infer(self, frame: AudioFrame) -> InferenceResult:
        mono = frame.mono()
        if mono.size == 0:
            spectral_centroid = 0.0
            rms = 0.0
        else:
            spectrum = np.abs(np.fft.rfft(mono * np.hanning(mono.size)))
            freqs = np.fft.rfftfreq(mono.size, d=1.0 / frame.sample_rate)
            total = float(np.sum(spectrum)) + 1e-12
            spectral_centroid = float(np.sum(freqs * spectrum) / total)
            rms = float(np.sqrt(np.mean(np.square(mono))))

        clarity = np.clip((2600.0 - spectral_centroid) / 2600.0, 0.0, 1.0) * 1.6
        warmth = np.clip((spectral_centroid - 1800.0) / 3600.0, 0.0, 1.0) * -0.8
        transient = np.clip((0.18 - rms) / 0.18, 0.0, 1.0) * 0.9

        return InferenceResult(
            clarity_gain_db=float(clarity),
            warmth_gain_db=float(warmth),
            transient_boost_db=float(transient),
            stereo_width=1.04 if rms < 0.35 else 1.0,
            backend=self.kind,
        )


class OnnxRuntimeBackend(InferenceBackend):
    """ONNX Runtime wrapper that prefers QNN on Snapdragon X."""

    def __init__(self, model_path: str, providers: Sequence[str], kind: BackendKind) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depends on optional package.
            raise RuntimeError("onnxruntime is not installed") from exc

        self.kind = kind
        self._session = ort.InferenceSession(model_path, providers=list(providers))
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [output.name for output in self._session.get_outputs()]

    def infer(self, frame: AudioFrame) -> InferenceResult:
        features = _extract_model_features(frame)
        outputs = self._session.run(self._output_names, {self._input_name: features})
        controls = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        padded = np.pad(controls, (0, max(0, 4 - controls.size)), constant_values=0.0)
        return InferenceResult(
            clarity_gain_db=float(np.clip(padded[0], 0.0, 3.0)),
            warmth_gain_db=float(np.clip(padded[1], -3.0, 3.0)),
            transient_boost_db=float(np.clip(padded[2], 0.0, 2.0)),
            stereo_width=float(np.clip(1.0 + padded[3], 0.85, 1.2)),
            backend=self.kind,
        )


def select_backend(model_path: str | None = None) -> InferenceBackend:
    """Select QNN, then DirectML, then CPU according to availability.

    Set ``SNAPDRAGON_AUDIO_BACKEND=cpu`` to force deterministic local testing.
    """

    requested = os.getenv("SNAPDRAGON_AUDIO_BACKEND", "auto").strip().lower()
    if requested == BackendKind.CPU.value or not model_path:
        return HeuristicCpuBackend()

    try:
        import onnxruntime as ort
    except ImportError:
        return HeuristicCpuBackend()

    available = set(ort.get_available_providers())
    if requested in {"auto", BackendKind.QNN.value} and "QNNExecutionProvider" in available:
        return OnnxRuntimeBackend(model_path, ["QNNExecutionProvider", "CPUExecutionProvider"], BackendKind.QNN)
    if requested in {"auto", BackendKind.DIRECTML.value} and "DmlExecutionProvider" in available:
        return OnnxRuntimeBackend(model_path, ["DmlExecutionProvider", "CPUExecutionProvider"], BackendKind.DIRECTML)
    return HeuristicCpuBackend()


def _extract_model_features(frame: AudioFrame) -> np.ndarray:
    mono = frame.mono()
    rms = np.sqrt(np.mean(np.square(mono)) + 1e-12)
    peak = np.max(np.abs(mono)) if mono.size else 0.0
    spectrum = np.abs(np.fft.rfft(mono * np.hanning(mono.size))) if mono.size else np.zeros(1)
    bands = np.array_split(spectrum, 8)
    band_energy = np.array([np.mean(band) if band.size else 0.0 for band in bands], dtype=np.float32)
    features = np.concatenate(([rms, peak], band_energy)).astype(np.float32)
    return features.reshape(1, -1)
