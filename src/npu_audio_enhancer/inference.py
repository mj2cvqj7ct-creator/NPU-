from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os

import numpy as np

from .audio import AudioBuffer


class InferenceBackend(str, Enum):
    QNN = "qnn"
    DIRECTML = "directml"
    CPU = "cpu"


@dataclass(frozen=True)
class AudioFeatures:
    backend: InferenceBackend
    clarity: float
    density: float
    bass_weight: float
    transient_risk: float

    def as_dict(self) -> dict[str, float]:
        return {
            "clarity": self.clarity,
            "density": self.density,
            "bass_weight": self.bass_weight,
            "transient_risk": self.transient_risk,
        }


class NpuFeatureExtractor:
    """Feature extraction boundary for Snapdragon X NPU acceleration.

    The production implementation can load an ONNX model through the QNN
    Execution Provider. This prototype keeps the same decision boundary and
    uses deterministic CPU features when QNN is not present.
    """

    def __init__(self, prefer_npu: bool = True) -> None:
        self.backend = self._select_backend(prefer_npu)

    def extract(self, audio: AudioBuffer) -> AudioFeatures:
        mono = audio.samples.mean(axis=1)
        if mono.size == 0:
            return AudioFeatures(self.backend, 0.5, 0.0, 0.0, 0.0)

        windowed = mono * np.hanning(mono.size)
        spectrum = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(mono.size, d=1.0 / audio.sample_rate_hz)
        total_energy = float(np.sum(spectrum) + 1e-9)

        bass = _band_energy(spectrum, freqs, 20.0, 180.0) / total_energy
        presence = _band_energy(spectrum, freqs, 1_500.0, 5_000.0) / total_energy
        air = _band_energy(spectrum, freqs, 8_000.0, 16_000.0) / total_energy

        rms = float(np.sqrt(np.mean(np.square(mono))))
        peak = float(np.max(np.abs(mono)))
        crest = peak / max(rms, 1e-6)

        return AudioFeatures(
            backend=self.backend,
            clarity=_clamp01(0.35 + presence * 3.4 + air * 1.8),
            density=_clamp01(rms * 3.2 + bass * 1.4),
            bass_weight=_clamp01(bass * 5.0),
            transient_risk=_clamp01((crest - 3.0) / 8.0),
        )

    @staticmethod
    def _select_backend(prefer_npu: bool) -> InferenceBackend:
        if prefer_npu and os.environ.get("SNAPDRAGON_X_QNN_AVAILABLE") == "1":
            return InferenceBackend.QNN
        if os.environ.get("DIRECTML_AVAILABLE") == "1":
            return InferenceBackend.DIRECTML
        return InferenceBackend.CPU


def _band_energy(spectrum: np.ndarray, freqs: np.ndarray, low_hz: float, high_hz: float) -> float:
    mask = (freqs >= low_hz) & (freqs < high_hz)
    return float(np.sum(spectrum[mask]))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
