from __future__ import annotations

import math
import os
from dataclasses import dataclass

from .dsp import AudioBuffer, EnhancementResult, analyze, enhance_buffer
from .profiles import EnhancementProfile


@dataclass(frozen=True)
class InferenceFeatures:
    loudness_lufs: float
    peak: float
    rms: float
    clipping_events: int
    stereo_correlation: float
    transient_density: float


@dataclass(frozen=True)
class NeuralGains:
    bass: float
    presence: float
    air: float
    width: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.bass, self.presence, self.air, self.width)


class InferenceBackend:
    name = "base"
    description = "base inference backend"

    def infer(self, buffer: AudioBuffer, profile: EnhancementProfile) -> NeuralGains:
        raise NotImplementedError

    def enhance(self, buffer: AudioBuffer, profile: EnhancementProfile) -> EnhancementResult:
        gains = self.infer(buffer, profile)
        return enhance_buffer(buffer, profile, gains.as_tuple(), self.name)


class HeuristicCpuBackend(InferenceBackend):
    name = "cpu-dsp"
    description = "deterministic CPU DSP fallback for systems without QNN"

    def infer(self, buffer: AudioBuffer, profile: EnhancementProfile) -> NeuralGains:
        features = extract_features(buffer)
        loudness_gap = max(0.0, profile.target_lufs - features.loudness_lufs)
        transient_need = min(1.0, features.transient_density * 4.0 + profile.transient_restore)
        clipping_penalty = 0.05 if features.clipping_events else 0.0

        bass = _clamp(1.0 + loudness_gap * 0.012 - clipping_penalty, 0.92, 1.10)
        presence = _clamp(1.0 + profile.presence_lift_db * 0.035 + transient_need * 0.04, 0.96, 1.16)
        air = _clamp(1.0 + profile.air_lift_db * 0.03 + transient_need * 0.03, 0.97, 1.14)
        width = _clamp(1.0 + (1.0 - features.stereo_correlation) * 0.08, 0.96, profile.max_stereo_width)
        return NeuralGains(bass=bass, presence=presence, air=air, width=width)


class QnnNpuBackend(InferenceBackend):
    name = "onnxruntime-qnn"
    description = "Snapdragon X NPU path through ONNX Runtime QNN Execution Provider"

    def __init__(self, model_path: str | None = None, force_available: bool | None = None) -> None:
        self.model_path = model_path or os.environ.get("NPU_AUDIO_QNN_MODEL")
        self.force_available = force_available

    def available(self) -> bool:
        if self.force_available is not None:
            return self.force_available
        return bool(self.model_path and os.path.exists(self.model_path))

    def infer(self, buffer: AudioBuffer, profile: EnhancementProfile) -> NeuralGains:
        if not self.available():
            raise RuntimeError("QNN model is not configured; set NPU_AUDIO_QNN_MODEL")

        # The production path will run ONNX Runtime QNN EP here. Until the model
        # is present, tests use the CPU backend through select_backend().
        return HeuristicCpuBackend().infer(buffer, profile)


@dataclass(frozen=True)
class BackendSelection:
    qnn_available: bool | None = None
    model_path: str | None = None


def detect_backend_selection(model_path: str | None = None) -> BackendSelection:
    candidate = model_path or os.environ.get("NPU_AUDIO_QNN_MODEL")
    return BackendSelection(
        qnn_available=bool(candidate and os.path.exists(candidate)),
        model_path=candidate,
    )


def select_backend(
    profile: EnhancementProfile,
    selection: BackendSelection | None = None,
    preference: str = "auto",
) -> InferenceBackend:
    preference = preference.strip().lower()
    if preference == "cpu":
        return HeuristicCpuBackend()

    selection = selection or BackendSelection()
    qnn = QnnNpuBackend(selection.model_path, selection.qnn_available)
    qnn_available = qnn.available() if selection.qnn_available is None else selection.qnn_available
    if preference == "qnn":
        return qnn if qnn_available else HeuristicCpuBackend()
    if preference != "auto":
        raise ValueError("backend must be one of: auto, qnn, cpu")
    if profile.target_backend == qnn.name and qnn_available:
        return qnn
    return HeuristicCpuBackend()


def create_backend(
    preference: str = "auto",
    selection: BackendSelection | None = None,
) -> InferenceBackend:
    preference = preference.strip().lower()
    if preference == "auto":
        return AutoBackend(selection)
    if preference in {"cpu", "qnn"}:
        return FixedPreferenceBackend(preference, selection)
    raise ValueError("backend must be one of: auto, qnn, cpu")


class FixedPreferenceBackend(InferenceBackend):
    def __init__(self, preference: str, selection: BackendSelection | None = None) -> None:
        self.preference = preference
        self.selection = selection

    def enhance(self, buffer: AudioBuffer, profile: EnhancementProfile) -> EnhancementResult:
        backend = select_backend(profile, self.selection, preference=self.preference)
        return backend.enhance(buffer, profile)

    def infer(self, buffer: AudioBuffer, profile: EnhancementProfile) -> NeuralGains:
        return select_backend(profile, self.selection, preference=self.preference).infer(buffer, profile)


class AutoBackend(InferenceBackend):
    name = "auto"
    description = "profile-aware backend selector"

    def __init__(self, selection: BackendSelection | None = None) -> None:
        self.selection = selection

    def enhance(self, buffer: AudioBuffer, profile: EnhancementProfile) -> EnhancementResult:
        backend = select_backend(profile, self.selection)
        return backend.enhance(buffer, profile)

    def infer(self, buffer: AudioBuffer, profile: EnhancementProfile) -> NeuralGains:
        return select_backend(profile, self.selection).infer(buffer, profile)


def extract_features(buffer: AudioBuffer) -> InferenceFeatures:
    metrics = analyze(buffer)
    return InferenceFeatures(
        loudness_lufs=metrics.loudness_lufs,
        peak=metrics.peak,
        rms=metrics.rms,
        clipping_events=metrics.clipping_events,
        stereo_correlation=_stereo_correlation(buffer),
        transient_density=_transient_density(buffer),
    )


def _stereo_correlation(buffer: AudioBuffer) -> float:
    if buffer.channels != 2 or not buffer.samples:
        return 1.0

    left = buffer.samples[0::2]
    right = buffer.samples[1::2]
    numerator = sum(l * r for l, r in zip(left, right))
    left_energy = math.sqrt(sum(l * l for l in left))
    right_energy = math.sqrt(sum(r * r for r in right))
    if left_energy == 0 or right_energy == 0:
        return 1.0
    return _clamp(numerator / (left_energy * right_energy), -1.0, 1.0)


def _transient_density(buffer: AudioBuffer) -> float:
    if buffer.frames < 2:
        return 0.0

    transitions = 0
    previous = buffer.samples[0]
    for sample in buffer.samples[1:]:
        if abs(sample - previous) > 0.35:
            transitions += 1
        previous = sample
    return transitions / max(1, len(buffer.samples) - 1)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
