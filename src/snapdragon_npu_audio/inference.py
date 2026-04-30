"""NPU provider selection and deterministic feature inference."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Protocol

from .types import AudioBuffer, clamp, peak, rms


class ProviderKind(str, Enum):
    QNN_NPU = "qnn_npu"
    DIRECTML = "directml"
    CPU = "cpu"


@dataclass(frozen=True)
class RuntimeProvider:
    kind: ProviderKind
    name: str
    reason: str
    accelerated: bool

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "kind": self.kind.value,
            "name": self.name,
            "reason": self.reason,
            "accelerated": self.accelerated,
        }


@dataclass(frozen=True)
class ProviderRequest:
    prefer_npu: bool = True
    available: tuple[str, ...] | None = None


@dataclass(frozen=True)
class MusicFeatures:
    bass_weight: float
    vocal_presence: float
    transient_density: float
    loudness: float


class InferenceProvider(Protocol):
    name: str

    def infer(self, buffer: AudioBuffer) -> MusicFeatures:
        """Return adaptive music features for one processing block."""


def select_provider(request: ProviderRequest | None = None) -> RuntimeProvider:
    request = request or ProviderRequest()
    available: Iterable[str]
    if request.available is None:
        env_value = os.getenv("SNAPDRAGON_AUDIO_PROVIDERS", "")
        available = [item.strip() for item in env_value.split(",") if item.strip()]
    else:
        available = request.available

    normalized = {item.lower(): item for item in available}
    if request.prefer_npu and "qnnexecutionprovider" in normalized:
        return RuntimeProvider(
            kind=ProviderKind.QNN_NPU,
            name=normalized["qnnexecutionprovider"],
            reason="Qualcomm QNN provider is available for Snapdragon X NPU inference.",
            accelerated=True,
        )
    if "dmlexecutionprovider" in normalized:
        return RuntimeProvider(
            kind=ProviderKind.DIRECTML,
            name=normalized["dmlexecutionprovider"],
            reason="DirectML provider is available as an accelerated fallback.",
            accelerated=True,
        )
    return RuntimeProvider(
        kind=ProviderKind.CPU,
        name="CPUExecutionProvider",
        reason="No accelerated provider was detected; using deterministic CPU fallback.",
        accelerated=False,
    )


class CpuFallbackProvider:
    """Feature extractor mirroring the future ONNX/QNN model contract."""

    name = "cpu_fallback"

    def infer(self, buffer: AudioBuffer) -> MusicFeatures:
        if not buffer.samples:
            return MusicFeatures(0.0, 0.0, 0.0, 0.0)

        mono = [sum(frame) / len(frame) for frame in buffer.samples]
        abs_mean = sum(abs(sample) for sample in mono) / len(mono)
        changes = [abs(mono[index] - mono[index - 1]) for index in range(1, len(mono))]
        change_mean = sum(changes) / max(len(changes), 1)
        zero_crossings = sum(
            1
            for index in range(1, len(mono))
            if (mono[index - 1] < 0.0 <= mono[index]) or (mono[index - 1] >= 0.0 > mono[index])
        )
        zero_crossing_rate = zero_crossings / max(len(mono) - 1, 1)

        bass_weight = clamp(abs_mean * 1.8, 0.0, 1.0)
        vocal_presence = clamp((1.0 - zero_crossing_rate) * 0.7 + rms(buffer.samples), 0.0, 1.0)
        transient_density = clamp(change_mean * 6.0, 0.0, 1.0)
        return MusicFeatures(
            bass_weight=bass_weight,
            vocal_presence=vocal_presence,
            transient_density=transient_density,
            loudness=peak(buffer.samples),
        )
