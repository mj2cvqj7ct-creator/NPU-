from __future__ import annotations

import importlib.util
import os
import platform
from dataclasses import dataclass
from typing import Iterable

from .dsp import AudioMetrics
from .profiles import ServiceProfile


PROVIDER_ALIASES = {
    "auto": "auto",
    "qnn": "qnn",
    "onnx_qnn": "qnn",
    "onnx-qnn": "qnn",
    "directml": "directml",
    "dml": "directml",
    "cpu": "cpu",
}


@dataclass(frozen=True)
class ProviderSelection:
    kind: str
    reason: str


@dataclass(frozen=True)
class InferenceTuning:
    presence_gain_db: float
    warmth_gain_db: float
    air_gain_db: float
    transient_restore: float
    stereo_width: float
    provider: ProviderSelection


class InferenceProvider:
    """Selects an inference provider and maps audio features to safe DSP targets."""

    def __init__(
        self,
        preferred_providers: Iterable[str] | None = None,
        force_provider: str | None = None,
    ) -> None:
        if force_provider is None:
            forced = os.getenv("SNAPDRAGON_AUDIO_PROVIDER")
            force_provider = forced if forced else None

        if force_provider is not None:
            provider = _normalize_provider(force_provider)
            self.provider = self._select_forced(provider)
            return

        providers = (
            [_normalize_provider(provider) for provider in preferred_providers]
            if preferred_providers is not None
            else ["qnn", "directml", "cpu"]
        )
        self.provider = self._select_available(providers)

    @property
    def name(self) -> str:
        return self.provider.kind

    def infer(
        self, metrics: AudioMetrics, sample_rate_hz: int, profile: ServiceProfile
    ) -> InferenceTuning:
        del sample_rate_hz
        density = min(1.0, metrics.crest_factor_db / 14.0)
        low_volume = max(0.0, min(1.0, (-18.0 - metrics.rms_db) / 30.0))
        clipping_pressure = max(0.0, min(1.0, (metrics.peak - 0.85) / 0.15))

        presence = profile.presence_gain_db * (0.65 + 0.35 * low_volume)
        warmth = profile.bass_gain_db * (0.75 + 0.25 * low_volume)
        air = profile.air_gain_db * (1.0 - 0.35 * clipping_pressure)
        transient = (
            profile.transient_restore
            * (1.0 - 0.45 * clipping_pressure)
            * (0.7 + 0.3 * density)
        )
        width = profile.stereo_width * (1.0 - 0.25 * clipping_pressure)

        if self.provider.kind == "cpu":
            transient *= 0.9
            width *= 0.98

        return InferenceTuning(
            presence_gain_db=round(presence, 3),
            warmth_gain_db=round(warmth, 3),
            air_gain_db=round(air, 3),
            transient_restore=round(max(0.0, transient), 3),
            stereo_width=round(max(0.85, min(1.2, width)), 3),
            provider=self.provider,
        )

    @staticmethod
    def _select_forced(provider: str) -> ProviderSelection:
        if provider == "auto":
            return InferenceProvider._select_available(["qnn", "directml", "cpu"])
        if provider == "qnn" and not _qnn_available():
            return ProviderSelection(
                "cpu",
                "QNN was forced but onnxruntime QNN support is unavailable; using CPU fallback.",
            )
        if provider == "directml" and not _directml_available():
            return ProviderSelection(
                "cpu",
                "DirectML was forced but is unavailable; using CPU fallback.",
            )
        return ProviderSelection(provider, f"{provider} provider forced by configuration.")

    @staticmethod
    def _select_available(providers: list[str]) -> ProviderSelection:
        for provider in providers:
            if provider == "auto":
                continue
            if provider == "qnn" and _qnn_available():
                return ProviderSelection("qnn", "ONNX Runtime QNN Execution Provider detected.")
            if provider == "directml" and _directml_available():
                return ProviderSelection("directml", "DirectML provider detected.")
            if provider == "cpu":
                return ProviderSelection("cpu", "Portable CPU fallback selected.")
        return ProviderSelection("cpu", "No requested accelerator was available; using CPU fallback.")


def create_inference_provider(preferred_provider: str | None = None) -> InferenceProvider:
    if preferred_provider is None or preferred_provider == "auto":
        return InferenceProvider()
    return InferenceProvider(force_provider=preferred_provider)


def _normalize_provider(provider: str) -> str:
    normalized = provider.strip().lower().replace(" ", "_")
    try:
        return PROVIDER_ALIASES[normalized]
    except KeyError as exc:
        choices = ", ".join(sorted(PROVIDER_ALIASES))
        raise ValueError(f"unknown inference provider {provider!r}; expected one of {choices}") from exc


def _qnn_available() -> bool:
    if platform.machine().lower() not in {"arm64", "aarch64"}:
        return False
    if importlib.util.find_spec("onnxruntime") is None:
        return False
    try:
        import onnxruntime as ort  # type: ignore
    except Exception:
        return False
    return "QNNExecutionProvider" in ort.get_available_providers()


def _directml_available() -> bool:
    if importlib.util.find_spec("onnxruntime") is None:
        return False
    try:
        import onnxruntime as ort  # type: ignore
    except Exception:
        return False
    return "DmlExecutionProvider" in ort.get_available_providers()
