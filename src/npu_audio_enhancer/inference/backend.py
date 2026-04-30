from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
import platform

from npu_audio_enhancer.dsp.frame import AudioFrame
from npu_audio_enhancer.dsp.loudness import rms


@dataclass(frozen=True)
class InferenceResult:
    backend_name: str
    neural_gain: float
    clarity_hint: float
    used_npu: bool


class SnapdragonNpuBackendSelector:
    """Selects the best available inference path without hard requiring QNN."""

    def __init__(self, prefer_npu: bool = True) -> None:
        self.prefer_npu = prefer_npu

    def select_backend_name(self) -> str:
        if self.prefer_npu and _looks_like_arm64() and _onnxruntime_available():
            return "onnxruntime-qnn"
        if _onnxruntime_available():
            return "onnxruntime-cpu"
        return "deterministic-cpu"

    def infer(self, frame: AudioFrame, service_profile: str) -> InferenceResult:
        backend_name = self.select_backend_name()
        loudness_energy = rms(frame)
        service_bias = _service_clarity_bias(service_profile)
        clarity_hint = _clamp((0.22 - loudness_energy) * 1.8 + service_bias, 0.0, 1.0)
        neural_gain = 1.0 + clarity_hint * 0.035
        return InferenceResult(
            backend_name=backend_name,
            neural_gain=neural_gain,
            clarity_hint=clarity_hint,
            used_npu=backend_name == "onnxruntime-qnn",
        )


def _onnxruntime_available() -> bool:
    return find_spec("onnxruntime") is not None


def _looks_like_arm64() -> bool:
    machine = platform.machine().lower()
    return machine in {"arm64", "aarch64"} or "arm" in machine


def _service_clarity_bias(service_profile: str) -> float:
    normalized = service_profile.lower().replace(" ", "-")
    if "youtube" in normalized:
        return 0.08
    if "spotify" in normalized:
        return 0.04
    if "apple" in normalized:
        return 0.02
    return 0.0


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
