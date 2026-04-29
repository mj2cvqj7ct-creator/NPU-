from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .dsp import (
    apply_dynamic_tilt_eq,
    apply_gain,
    apply_true_peak_limiter,
    estimate_loudness_lufs,
    normalize_loudness,
)
from .inference import BackendStatus, InferenceBackend, select_backend
from .models import AudioFrame, EnhancementSettings, InferenceDecision, ServiceProfile


SERVICE_SETTINGS: dict[ServiceProfile, EnhancementSettings] = {
    ServiceProfile.SPOTIFY: EnhancementSettings(
        target_loudness_lufs=-15.0,
        max_gain_db=9.0,
        bass_tilt_db=0.6,
        presence_tilt_db=1.2,
        stereo_width=1.04,
    ),
    ServiceProfile.APPLE_MUSIC: EnhancementSettings(
        target_loudness_lufs=-17.0,
        max_gain_db=6.0,
        bass_tilt_db=0.3,
        presence_tilt_db=0.7,
        stereo_width=1.02,
    ),
    ServiceProfile.YOUTUBE_MUSIC: EnhancementSettings(
        target_loudness_lufs=-14.5,
        max_gain_db=8.0,
        bass_tilt_db=0.2,
        presence_tilt_db=1.4,
        stereo_width=1.01,
    ),
    ServiceProfile.GENERIC: EnhancementSettings(
        target_loudness_lufs=-16.0,
        max_gain_db=7.0,
        bass_tilt_db=0.3,
        presence_tilt_db=0.9,
        stereo_width=1.02,
    ),
}


@dataclass(frozen=True)
class EnhancementResult:
    frame: AudioFrame
    backend: BackendStatus
    decision: InferenceDecision
    input_loudness_lufs: float
    output_peak: float


class AudioEnhancementPipeline:
    """Low-latency PCM enhancement chain with a pluggable NPU inference stage."""

    def __init__(
        self,
        *,
        service: ServiceProfile = ServiceProfile.GENERIC,
        backend: InferenceBackend | None = None,
        settings: EnhancementSettings | None = None,
    ) -> None:
        self.backend = backend or select_backend()
        self.service = service
        self.settings = settings or SERVICE_SETTINGS[service]

    @classmethod
    def for_service(cls, service: ServiceProfile | str) -> "AudioEnhancementPipeline":
        resolved = resolve_service_profile(service)
        return cls(service=resolved)

    def process(self, frame: AudioFrame) -> EnhancementResult:
        normalized = normalize_loudness(
            frame,
            target_lufs=self.settings.target_loudness_lufs,
            max_gain_db=self.settings.max_gain_db,
        )

        features = self.backend.analyze(normalized)
        decision = self.backend.decide(features, self.service, self.settings)

        equalized = apply_dynamic_tilt_eq(
            normalized,
            bass_db=decision.bass_boost_db,
            presence_db=decision.clarity_boost_db,
            stereo_width=self.settings.stereo_width,
        )
        compensated = apply_gain(equalized, decision.low_volume_compensation_db)
        limited = apply_true_peak_limiter(
            compensated,
            ceiling_dbfs=_linear_to_dbfs(self.settings.limiter_ceiling),
        )

        return EnhancementResult(
            frame=limited,
            backend=self.backend.status,
            decision=decision,
            input_loudness_lufs=estimate_loudness_lufs(frame),
            output_peak=limited.peak,
        )


def resolve_service_profile(service: ServiceProfile | str | None) -> ServiceProfile:
    if isinstance(service, ServiceProfile):
        return service
    if not service:
        return ServiceProfile.GENERIC

    normalized = service.casefold().replace(" ", "_").replace("-", "_")
    for candidate in ServiceProfile:
        if normalized == candidate.value:
            return candidate
    return ServiceProfile.GENERIC


def process_frames(
    frames: Iterable[AudioFrame],
    *,
    service: ServiceProfile | str | None = None,
    pipeline: AudioEnhancementPipeline | None = None,
) -> Iterable[EnhancementResult]:
    enhancer = pipeline or AudioEnhancementPipeline(service=resolve_service_profile(service))
    for frame in frames:
        yield enhancer.process(frame)


def _linear_to_dbfs(value: float) -> float:
    # Importing from dsp would expose an implementation detail; this stays local.
    import math

    return 20.0 * math.log10(value)
