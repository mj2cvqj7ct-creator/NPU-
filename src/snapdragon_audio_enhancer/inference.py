from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, Sequence


class BackendKind(str, Enum):
    """Execution targets ordered by preference for Snapdragon X systems."""

    QNN_NPU = "qnn_npu"
    DIRECTML = "directml"
    CPU = "cpu"


@dataclass(frozen=True)
class RuntimeCapabilities:
    """Runtime feature flags discovered by the host application."""

    is_arm64: bool
    has_snapdragon_x_npu: bool
    has_qnn_execution_provider: bool
    has_directml: bool


@dataclass(frozen=True)
class InferenceConfig:
    """Real-time model constraints for the audio enhancement path."""

    sample_rate_hz: int = 48_000
    frame_size: int = 480
    max_frame_latency_ms: float = 20.0
    enhancement_mix: float = 0.35

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.frame_size <= 0:
            raise ValueError("frame_size must be positive")
        if not 0.0 <= self.enhancement_mix <= 1.0:
            raise ValueError("enhancement_mix must be between 0.0 and 1.0")


@dataclass(frozen=True)
class InferenceResult:
    """DSP control values produced by the NPU or fallback path."""

    backend: BackendKind
    bass_weight: float = 1.0
    presence_weight: float = 1.0
    width_weight: float = 1.0


class AudioModel(Protocol):
    """Minimal interface implemented by ONNX/QNN or fallback models."""

    def infer(self, frame: Sequence[float], features: dict[str, float]) -> list[float]:
        """Return an enhanced mono feature/control frame."""


class PresenceEnhancementModel:
    """Deterministic fallback that emulates a light clarity model.

    The fallback does not claim to restore missing source information. It gives
    the DSP chain stable control data when the real QNN-backed model is absent.
    """

    def __init__(self, config: InferenceConfig | None = None) -> None:
        self.config = config or InferenceConfig()

    def infer(self, frame: Sequence[float], features: dict[str, float]) -> list[float]:
        if not frame:
            return []

        density = min(1.0, max(0.0, features.get("spectral_density", 0.0)))
        clarity_gain = 1.0 + (0.08 * density)
        mixed: list[float] = []
        previous = 0.0

        for sample in frame:
            transient = sample - previous
            enhanced = sample + (0.18 * transient)
            mixed_sample = (
                sample * (1.0 - self.config.enhancement_mix)
                + enhanced * self.config.enhancement_mix * clarity_gain
            )
            mixed.append(max(-1.0, min(1.0, mixed_sample)))
            previous = sample

        return mixed


def choose_backend(capabilities: RuntimeCapabilities) -> BackendKind:
    """Pick the fastest safe execution provider available on the host."""

    if (
        capabilities.is_arm64
        and capabilities.has_snapdragon_x_npu
        and capabilities.has_qnn_execution_provider
    ):
        return BackendKind.QNN_NPU
    if capabilities.has_directml:
        return BackendKind.DIRECTML
    return BackendKind.CPU


class PassthroughEngine:
    """Safe default when no NPU-backed model has been configured yet."""

    backend = BackendKind.CPU

    def enhance(self, analysis: Any) -> InferenceResult:
        return InferenceResult(backend=self.backend)


@dataclass
class InferenceEngine:
    """Small boundary around model execution and backend selection."""

    capabilities: RuntimeCapabilities
    model: AudioModel
    config: InferenceConfig = InferenceConfig()

    @property
    def backend(self) -> BackendKind:
        return choose_backend(self.capabilities)

    def enhance(self, analysis: Any) -> InferenceResult:
        """Convert frame analysis into bounded DSP controls.

        Real deployments can replace ``model`` with an ONNX/QNN model while
        preserving this narrow output contract for the real-time DSP chain.
        """

        low_energy = max(0.0, float(getattr(analysis, "low_band_energy", 0.0)))
        high_energy = max(0.0, float(getattr(analysis, "high_band_energy", 0.0)))
        total_energy = low_energy + high_energy
        density = 0.0 if total_energy == 0.0 else high_energy / total_energy

        control_frame = self.model.infer(
            [density],
            {
                "spectral_density": density,
                "rms_dbfs": float(getattr(analysis, "rms_dbfs", -120.0)),
                "peak": float(getattr(analysis, "peak", 0.0)),
            },
        )
        model_control = control_frame[0] if control_frame else density

        bass_weight = _clamp(1.1 - density, 0.75, 1.2)
        presence_weight = _clamp(0.85 + model_control, 0.85, 1.25)
        width_weight = _clamp(1.0 - abs(float(getattr(analysis, "channel_balance", 0.0))) * 0.2, 0.85, 1.1)

        return InferenceResult(
            backend=self.backend,
            bass_weight=bass_weight,
            presence_weight=presence_weight,
            width_weight=width_weight,
        )

    def enhance_control_frame(
        self, frame: Sequence[float], features: dict[str, float]
    ) -> list[float]:
        if len(frame) > self.config.frame_size:
            raise ValueError("frame exceeds configured low-latency frame size")
        return self.model.infer(frame, features)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))
