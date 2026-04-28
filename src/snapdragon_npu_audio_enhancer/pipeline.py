"""Service-agnostic audio enhancement pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

from .audio import AudioBuffer
from .dsp import (
    EnhancementProfile,
    adjust_stereo_width,
    normalize_loudness,
    tone_shape,
    true_peak_limiter,
)
from .inference import AudioEnhancementModel, HeuristicEnhancementModel, InferenceBackend


@dataclass(frozen=True)
class EnhancementConfig:
    """Controls for the offline prototype.

    The service name is only metadata. Audio from Spotify, Apple Music, and
    YouTube Music is treated the same once it reaches the OS PCM stream.
    """

    service: str = "generic"
    profile: EnhancementProfile = field(default_factory=EnhancementProfile)
    backend: InferenceBackend = InferenceBackend.CPU


@dataclass(frozen=True)
class EnhancementResult:
    audio: AudioBuffer
    input_peak: float
    output_peak: float
    input_rms: float
    output_rms: float
    service: str
    backend: InferenceBackend


class EnhancementPipeline:
    """Offline stand-in for the planned realtime WASAPI/APO processing chain."""

    def __init__(
        self,
        config: EnhancementConfig | None = None,
        model: AudioEnhancementModel | None = None,
    ) -> None:
        self.config = config or EnhancementConfig()
        self.model = model or HeuristicEnhancementModel()

    def process(self, audio: AudioBuffer) -> EnhancementResult:
        controls = self.model.infer(audio)
        profile = self.config.profile

        processed = normalize_loudness(
            audio,
            EnhancementProfile(
                target_rms_dbfs=controls.loudness_target_lufs,
                max_gain_db=profile.max_gain_db,
                bass_boost_db=profile.bass_boost_db,
                presence_boost_db=profile.presence_boost_db,
                air_boost_db=profile.air_boost_db,
                stereo_width=profile.stereo_width,
                limiter_ceiling_dbfs=profile.limiter_ceiling_dbfs,
            ),
        )
        processed = tone_shape(
            processed,
            profile,
            clarity=max(0.0, controls.clarity_db / max(profile.presence_boost_db, 0.01)),
            warmth=max(0.0, controls.warmth_db / max(profile.bass_boost_db, 0.01)),
            air=0.7,
        )
        processed = adjust_stereo_width(processed, profile.stereo_width * controls.stereo_width)
        processed = true_peak_limiter(processed, profile.limiter_ceiling_dbfs)

        return EnhancementResult(
            audio=processed,
            input_peak=audio.peak,
            output_peak=processed.peak,
            input_rms=audio.rms,
            output_rms=processed.rms,
            service=self.config.service,
            backend=self.config.backend,
        )


def enhance_samples(
    audio: AudioBuffer,
    config: EnhancementConfig | None = None,
    model: AudioEnhancementModel | None = None,
) -> EnhancementResult:
    """Convenience API for tests and future capture/render adapters."""

    return EnhancementPipeline(config=config, model=model).process(audio)
