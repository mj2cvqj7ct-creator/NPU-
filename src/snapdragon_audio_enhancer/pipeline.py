from __future__ import annotations

from dataclasses import dataclass

from .audio_types import AudioBuffer, EnhancementTelemetry
from .dsp import (
    apply_soft_limiter,
    apply_stereo_width,
    apply_tone_shape,
    apply_transient_restore,
    normalize_loudness,
)
from .inference import InferenceBackend, select_backend
from .service_profiles import MusicService, ServiceProfile, get_profile


@dataclass(frozen=True)
class PipelineStats:
    backend_name: str
    peak: float
    rms: float
    applied_gain_db: float
    limiter_reductions: int


class EnhancementPipeline:
    """PCM enhancer with an explicit Snapdragon X NPU inference boundary."""

    def __init__(
        self,
        profile: ServiceProfile | None = None,
        backend: InferenceBackend | None = None,
        backend_preference: str = "auto",
        qnn_enabled: bool = False,
        model_path: str | None = None,
    ) -> None:
        self.profile = profile or get_profile(MusicService.SPOTIFY)
        prefer_npu = qnn_enabled and backend_preference in ("auto", "qnn")
        self.backend = backend or select_backend(model_path=model_path, prefer_npu=prefer_npu)
        self.last_stats = PipelineStats(
            backend_name=self.backend.name,
            peak=0.0,
            rms=0.0,
            applied_gain_db=0.0,
            limiter_reductions=0,
        )

    def process(self, buffer: AudioBuffer) -> AudioBuffer:
        controls = self.backend.infer(buffer, self.profile)
        enhanced, gain_db = normalize_loudness(buffer, self.profile.target_loudness_lufs)
        enhanced = apply_tone_shape(
            enhanced,
            bass_gain_db=self.profile.bass_gain_db + controls.warmth * 6.0,
            presence_gain_db=self.profile.presence_gain_db + controls.clarity * 6.0,
            air_gain_db=self.profile.air_gain_db + controls.clarity * 3.0,
        )
        enhanced = apply_transient_restore(
            enhanced,
            amount=min(0.28, self.profile.transient_restore + controls.transient_restore),
        )
        enhanced = apply_stereo_width(enhanced, self.profile.stereo_width + controls.stereo_width)
        enhanced, limiter_reductions = apply_soft_limiter(enhanced, self.profile.limiter_ceiling)

        self.last_stats = PipelineStats(
            backend_name=controls.backend_name,
            peak=enhanced.peak,
            rms=enhanced.rms,
            applied_gain_db=gain_db,
            limiter_reductions=limiter_reductions,
        )
        return enhanced

    def process_with_telemetry(self, buffer: AudioBuffer) -> tuple[AudioBuffer, EnhancementTelemetry]:
        input_peak = buffer.peak
        input_rms = buffer.rms
        enhanced = self.process(buffer)
        telemetry = EnhancementTelemetry(
            backend=self.last_stats.backend_name,
            service=self.profile.service.value,
            sample_rate=buffer.sample_rate,
            input_peak=input_peak,
            output_peak=enhanced.peak,
            input_rms=input_rms,
            output_rms=enhanced.rms,
            gain_db=self.last_stats.applied_gain_db,
            limiter_reductions=self.last_stats.limiter_reductions,
            used_npu=self.last_stats.backend_name == "onnxruntime-qnn",
        )
        return enhanced, telemetry

    @staticmethod
    def _db_control(value: float) -> float:
        return max(0.0, min(0.18, value / 24.0))


AudioEnhancementPipeline = EnhancementPipeline
