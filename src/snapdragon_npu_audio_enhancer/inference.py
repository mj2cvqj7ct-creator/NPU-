"""Inference provider selection for Snapdragon X audio enhancement.

This module keeps vendor-specific acceleration behind a small interface. The
prototype uses a deterministic feature estimator so the DSP path can be tested
on any machine, while production builds can attach an ONNX Runtime session with
the QNN execution provider on ARM64 Snapdragon X systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import importlib.util
import math
import platform
from typing import Protocol

from .audio_frame import AudioFrame


class Provider(StrEnum):
    QNN = "qnn"
    DIRECTML = "directml"
    CPU = "cpu"


@dataclass(frozen=True)
class InferenceConfig:
    prefer_npu: bool = True
    model_path: str | None = None
    hardware_name: str | None = None
    available_providers: tuple[Provider, ...] = (Provider.CPU,)


class InferenceRouter:
    def __init__(self, config: InferenceConfig | None = None) -> None:
        self.config = config or InferenceConfig(
            available_providers=detect_available_providers()
        )
        self._selected_provider = self.select_provider()
        self._provider = self._build_provider()

    def select_provider(self) -> Provider:
        hardware = (self.config.hardware_name or platform.platform()).lower()
        providers = self.config.available_providers
        if (
            self.config.prefer_npu
            and Provider.QNN in providers
            and ("snapdragon" in hardware or "qualcomm" in hardware or is_snapdragon_arm64())
        ):
            return Provider.QNN
        if Provider.DIRECTML in providers:
            return Provider.DIRECTML
        return Provider.CPU

    def estimate_features(self, frame: AudioFrame) -> EnhancementFeatures:
        return self._provider.infer(frame)

    def _build_provider(self) -> InferenceProvider:
        if self._selected_provider == Provider.QNN and self.config.model_path:
            try:
                return OnnxQnnProvider(self.config.model_path)
            except Exception:
                # Runtime/provider availability varies by device image; keep playback safe.
                pass
        return CpuFeatureProvider()


@dataclass(frozen=True)
class EnhancementFeatures:
    """Frame-level model output used to steer the DSP chain."""

    clarity: float
    warmth: float
    transient: float
    stereo_width: float
    confidence: float


class InferenceProvider(Protocol):
    """Small interface shared by NPU and CPU inference providers."""

    name: str

    def infer(self, frame: AudioFrame) -> EnhancementFeatures:
        """Return enhancement controls for one audio frame."""


class CpuFeatureProvider:
    """Portable fallback estimator used when no NPU runtime is available."""

    name = "cpu-feature-estimator"

    def infer(self, frame: AudioFrame) -> EnhancementFeatures:
        mono = frame.mono()
        if not mono:
            return EnhancementFeatures(0.0, 0.0, 0.0, 0.0, 0.0)

        low = _goertzel_band_energy(mono, frame.sample_rate, (80.0, 160.0, 240.0))
        presence = _goertzel_band_energy(mono, frame.sample_rate, (2_000.0, 3_000.0, 4_500.0))
        air = _goertzel_band_energy(mono, frame.sample_rate, (8_000.0, 10_000.0, 12_000.0))
        total = low + presence + air + 1e-9

        left, right = frame.stereo_samples
        side_power = _mean_square([l_sample - r_sample for l_sample, r_sample in zip(left, right, strict=True)])
        mid_power = _mean_square([l_sample + r_sample for l_sample, r_sample in zip(left, right, strict=True)])
        width = math.sqrt(side_power) / (math.sqrt(mid_power) + 1e-9)

        derivative = [mono[index] - mono[index - 1] for index in range(1, len(mono))]
        transient = math.sqrt(_mean_square(derivative)) / (frame.rms + 1e-9) if derivative else 0.0

        return EnhancementFeatures(
            clarity=_clip((presence / total) * 3.2 + (air / total) * 1.4, 0.0, 1.0),
            warmth=_clip((low / total) * 2.6, 0.0, 1.0),
            transient=_clip(transient * 0.22, 0.0, 1.0),
            stereo_width=_clip(width, 0.0, 1.0),
            confidence=0.45,
        )


class OnnxQnnProvider:
    """Thin ONNX Runtime QNN provider wrapper for Snapdragon X deployments."""

    name = "onnxruntime-qnn"

    def __init__(self, model_path: str) -> None:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        if "QNNExecutionProvider" not in providers:
            raise RuntimeError("ONNX Runtime QNNExecutionProvider is not available")

        self._session = ort.InferenceSession(
            model_path,
            providers=["QNNExecutionProvider", "CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    def infer(self, frame: AudioFrame) -> EnhancementFeatures:
        import numpy as np

        model_input = np.asarray(frame.samples, dtype=np.float32)[None, :, :]
        output = self._session.run([self._output_name], {self._input_name: model_input})[0]
        values = np.asarray(output, dtype=np.float32).reshape(-1)
        padded = np.pad(values[:5], (0, max(0, 5 - values[:5].size)), constant_values=0.0)
        return EnhancementFeatures(
            clarity=_clip(float(padded[0]), 0.0, 1.0),
            warmth=_clip(float(padded[1]), 0.0, 1.0),
            transient=_clip(float(padded[2]), 0.0, 1.0),
            stereo_width=_clip(float(padded[3]), 0.0, 1.0),
            confidence=_clip(float(padded[4]), 0.0, 1.0),
        )


def is_snapdragon_arm64() -> bool:
    """Best-effort detection for ARM64 Windows Snapdragon-class PCs."""

    machine = platform.machine().lower()
    processor = platform.processor().lower()
    platform_text = platform.platform().lower()
    is_arm64 = machine in {"arm64", "aarch64"} or "arm" in processor
    looks_snapdragon = "snapdragon" in processor or "qualcomm" in platform_text
    return is_arm64 and looks_snapdragon


def detect_available_providers() -> tuple[Provider, ...]:
    providers = [Provider.CPU]
    if importlib.util.find_spec("onnxruntime") is None:
        return tuple(providers)

    try:
        import onnxruntime as ort
    except Exception:
        return tuple(providers)

    ort_providers = set(ort.get_available_providers())
    if "DmlExecutionProvider" in ort_providers:
        providers.insert(0, Provider.DIRECTML)
    if "QNNExecutionProvider" in ort_providers:
        providers.insert(0, Provider.QNN)
    return tuple(dict.fromkeys(providers))


def choose_provider(model_path: str | None = None, prefer_npu: bool = True) -> InferenceProvider:
    """Choose QNN on Snapdragon ARM64 when possible, then fall back to CPU."""
    config = InferenceConfig(
        prefer_npu=prefer_npu,
        model_path=model_path,
        available_providers=detect_available_providers(),
    )
    hardware = (config.hardware_name or platform.platform()).lower()
    qnn_selected = (
        config.prefer_npu
        and Provider.QNN in config.available_providers
        and ("snapdragon" in hardware or "qualcomm" in hardware or is_snapdragon_arm64())
    )
    if qnn_selected and model_path:
        try:
            return OnnxQnnProvider(model_path)
        except Exception:
            # Runtime/provider availability varies by device image; keep playback safe.
            pass

    return CpuFeatureProvider()


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _mean_square(samples: list[float]) -> float:
    if not samples:
        return 0.0
    return sum(sample * sample for sample in samples) / len(samples)


def _goertzel_band_energy(samples: list[float], sample_rate: int, frequencies: tuple[float, ...]) -> float:
    if not samples:
        return 0.0
    total = 0.0
    count = len(samples)
    for frequency in frequencies:
        normalized = frequency / sample_rate
        coeff = 2.0 * math.cos(2.0 * math.pi * normalized)
        s_prev = 0.0
        s_prev2 = 0.0
        for sample in samples:
            s_current = sample + coeff * s_prev - s_prev2
            s_prev2 = s_prev
            s_prev = s_current
        total += s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2
    return total / count
