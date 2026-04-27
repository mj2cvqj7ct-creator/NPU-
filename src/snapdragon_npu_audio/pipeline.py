"""End-to-end audio enhancement pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .audio import AudioBuffer
from .dsp import (
    CompressorSettings,
    EnhancementMetrics,
    LimiterSettings,
    analyze,
    apply_dynamic_eq,
    apply_multiband_compression,
    apply_stereo_width_guard,
    loudness_normalize,
    true_peak_limit,
)
from .npu import BackendKind, NpuAssistModel


@dataclass(frozen=True)
class EnhancementConfig:
    """Runtime controls for conservative music enhancement."""

    target_lufs: float = -16.0
    max_gain_db: float = 8.0
    limiter_ceiling_db: float = -1.0
    stereo_width: float = 1.08
    enable_npu_assist: bool = True
    preferred_backend: BackendKind | str | None = None


@dataclass(frozen=True)
class EnhancementReport:
    """Measurements and decisions from one processing pass."""

    input_metrics: EnhancementMetrics
    output_metrics: EnhancementMetrics
    backend: BackendKind
    predicted_controls: Mapping[str, float]


class SnapdragonAudioEnhancer:
    """Enhance PCM output from music apps without touching app internals."""

    def __init__(
        self,
        config: EnhancementConfig | None = None,
        model: NpuAssistModel | None = None,
    ) -> None:
        self.config = config or EnhancementConfig()
        preferred_backend = (
            BackendKind(self.config.preferred_backend)
            if isinstance(self.config.preferred_backend, str)
            else self.config.preferred_backend
        )
        self.backend = preferred_backend or (model.backend if model else NpuAssistModel().backend)
        self.model = model or NpuAssistModel(self.backend)

    def process(self, audio: AudioBuffer) -> tuple[AudioBuffer, EnhancementReport]:
        """Run rule-based DSP with optional NPU-assisted control prediction."""

        source = audio.ensure_stereo().as_float32()
        input_metrics = analyze(source)

        if self.config.enable_npu_assist:
            controls = self.model.predict_controls(input_metrics)
        else:
            controls = {}

        enhanced = loudness_normalize(
            source,
            target_lufs=self.config.target_lufs,
            max_gain_db=self.config.max_gain_db,
        )
        enhanced = apply_dynamic_eq(enhanced, controls)
        enhanced = apply_multiband_compression(
            enhanced,
            CompressorSettings(
                threshold_db=-13.0,
                ratio=1.8,
                makeup_gain_db=controls.get("clarity_db", 0.0) * 0.25,
            ),
        )
        enhanced = apply_stereo_width_guard(
            enhanced,
            width=self.config.stereo_width + controls.get("stereo_width_delta", 0.0),
        )
        enhanced = true_peak_limit(
            enhanced,
            LimiterSettings(ceiling_db=self.config.limiter_ceiling_db),
        )

        output_metrics = analyze(enhanced)
        report = EnhancementReport(
            input_metrics=input_metrics,
            output_metrics=output_metrics,
            backend=self.backend,
            predicted_controls=controls,
        )
        return enhanced, report
