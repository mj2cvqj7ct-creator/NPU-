from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .dsp import AudioFeatures, EnhancementControls
from .profiles import ServiceProfile


class InferenceBackend(Protocol):
    """Maps audio features to enhancement controls."""

    name: str

    def infer(self, features: AudioFeatures, profile: ServiceProfile) -> EnhancementControls:
        ...


@dataclass
class HeuristicNpuSurrogate:
    """Deterministic stand-in for the compact model intended for Snapdragon X NPU.

    The production path should export this control policy as ONNX and execute it
    through ONNX Runtime QNN or Qualcomm AI Engine Direct. Keeping this backend
    deterministic makes offline tests and tuning reproducible on non-NPU hosts.
    """

    name: str = "heuristic-npu-surrogate"

    def infer(self, features: AudioFeatures, profile: ServiceProfile) -> EnhancementControls:
        density = float(np.clip(features.spectral_flux * 3.0 + features.mid_band_energy, 0.0, 1.0))
        fatigue_risk = float(np.clip(features.high_band_energy * 10.0 + max(features.peak_db + 1.0, 0.0), 0.0, 1.0))
        low_deficit = float(np.clip(profile.low_band_reference - features.low_band_energy, -0.2, 0.35))
        vocal_deficit = float(np.clip(profile.vocal_reference - features.vocal_band_energy, -0.15, 0.3))

        bass = low_deficit * 10.0 * profile.bass_weight
        presence = vocal_deficit * 8.0 * profile.vocal_clarity_weight
        air = (0.12 - features.high_band_energy) * 10.0 * profile.air_weight
        transient = profile.transient_weight * (0.8 - density) * (1.0 - min(features.clipping_ratio * 250.0, 0.7))

        if fatigue_risk > 0.65:
            presence -= 0.7 * fatigue_risk
            air -= 0.9 * fatigue_risk
            transient *= 0.75

        return EnhancementControls(
            pre_gain_db=float(np.clip(profile.target_loudness_db - features.rms_db, -4.5, 3.0)),
            bass_gain_db=float(np.clip(bass, -1.5, 3.5)),
            presence_gain_db=float(np.clip(presence, -1.5, 2.5)),
            air_gain_db=float(np.clip(air, -1.5, 2.0)),
            stereo_width=float(np.clip(1.0 + profile.width_weight * (0.16 - features.side_energy), 0.92, 1.18)),
            transient_restore=float(np.clip(transient, 0.0, 0.45)),
            compressor_threshold_db=float(profile.compressor_threshold_db),
            compressor_ratio=float(profile.compressor_ratio),
            limiter_ceiling_db=float(profile.limiter_ceiling_db),
        )


@dataclass
class OnnxQnnBackend:
    """Optional ONNX Runtime backend configured for Snapdragon X QNN execution."""

    model_path: Path
    providers: tuple[str, ...] = ("QNNExecutionProvider", "CPUExecutionProvider")
    name: str = "onnx-qnn"

    def __post_init__(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError("onnxruntime is required for the ONNX QNN backend") from exc

        available = set(ort.get_available_providers())
        selected = [provider for provider in self.providers if provider in available]
        if not selected:
            selected = ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(self.model_path), providers=selected)
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    def infer(self, features: AudioFeatures, profile: ServiceProfile) -> EnhancementControls:
        vector = np.asarray([features.as_vector(profile)], dtype=np.float32)
        output = np.asarray(self._session.run([self._output_name], {self._input_name: vector})[0]).reshape(-1)
        return controls_from_vector(output, profile)


def controls_from_vector(values: np.ndarray, profile: ServiceProfile) -> EnhancementControls:
    """Decode a compact model output vector into bounded controls."""

    padded = np.zeros(8, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    padded[: min(values.size, padded.size)] = values[: padded.size]

    return EnhancementControls(
        pre_gain_db=float(np.clip(padded[0], -6.0, 4.0)),
        bass_gain_db=float(np.clip(padded[1], -2.5, 4.0)),
        presence_gain_db=float(np.clip(padded[2], -2.5, 3.0)),
        air_gain_db=float(np.clip(padded[3], -2.0, 2.5)),
        stereo_width=float(np.clip(1.0 + padded[4], 0.85, 1.25)),
        transient_restore=float(np.clip(padded[5], 0.0, 0.5)),
        compressor_threshold_db=float(np.clip(padded[6] or profile.compressor_threshold_db, -28.0, -10.0)),
        compressor_ratio=float(np.clip(padded[7] or profile.compressor_ratio, 1.0, 3.0)),
        limiter_ceiling_db=profile.limiter_ceiling_db,
    )


def build_backend(model_path: str | Path | None = None) -> InferenceBackend:
    """Build the best available backend without making NPU support mandatory."""

    configured = model_path or os.getenv("SNAPDRAGON_NPU_AUDIO_MODEL")
    if configured:
        try:
            return OnnxQnnBackend(Path(configured))
        except Exception:
            # The enhancement pipeline must remain usable when QNN/ORT is absent.
            pass
    return HeuristicNpuSurrogate()


def dump_model_contract(path: str | Path) -> None:
    """Write the feature/control schema expected by an ONNX NPU model."""

    contract = {
        "input": [
            "rms_db",
            "peak_db",
            "spectral_centroid_hz",
            "low_band_energy",
            "mid_band_energy",
            "vocal_band_energy",
            "high_band_energy",
            "spectral_flux",
            "clipping_ratio",
            "stereo_correlation",
            "side_energy",
            "profile_target_loudness_db",
            "profile_bass_weight",
            "profile_vocal_clarity_weight",
            "profile_air_weight",
            "profile_width_weight",
            "profile_transient_weight",
        ],
        "output": [
            "pre_gain_db",
            "bass_gain_db",
            "presence_gain_db",
            "air_gain_db",
            "stereo_width_delta",
            "transient_gain_db",
            "compressor_threshold_db",
            "compressor_ratio",
        ],
    }
    Path(path).write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
