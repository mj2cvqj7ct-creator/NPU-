"""Frame-level audio enhancement pipeline.

The pipeline operates only on PCM samples already emitted by the operating
system audio stack. Service labels select local tuning profiles; they do not
inspect or modify Spotify, Apple Music, or YouTube Music internals.
"""

from __future__ import annotations

from dataclasses import dataclass

from .dsp import (
    AudioFeatures,
    EnhancementSettings,
    StereoFrame,
    apply_loudness_gain,
    apply_stereo_width,
    apply_tone_shaping,
    apply_transient_restore,
    apply_true_peak_limiter,
    extract_features,
    measure_frame,
)
from .npu import EnhancementControls, HeuristicNpuModel, NpuEnhancementModel
from .profiles import EnhancementProfile, get_profile


@dataclass(frozen=True)
class PipelineResult:
    """Enhanced frame plus observability data for AB tests and telemetry."""

    frame: StereoFrame
    features: AudioFeatures
    applied_settings: EnhancementSettings


class AudioEnhancementPipeline:
    """Low-latency reference chain for Snapdragon X NPU audio experiments."""

    def __init__(
        self,
        service: str = "generic",
        npu_model: NpuEnhancementModel | None = None,
    ) -> None:
        self.profile: EnhancementProfile = get_profile(service)
        self.npu_model = npu_model or HeuristicNpuModel()

    def process_frame(self, frame: StereoFrame) -> PipelineResult:
        """Enhance one 10-20 ms stereo frame and preserve limiter invariants."""

        metrics = measure_frame(frame)
        features = extract_features(frame, metrics)
        controls = self.npu_model.infer(features)
        settings = self._settings_from_controls(controls)

        enhanced = apply_loudness_gain(frame, metrics, settings)
        enhanced = apply_tone_shaping(
            enhanced,
            settings,
            clarity_amount=controls.clarity,
            warmth_amount=controls.warmth,
        )
        enhanced = apply_transient_restore(
            enhanced,
            amount=min(self.profile.transient_restore, controls.transient_restore),
        )
        enhanced = apply_stereo_width(enhanced, settings, metrics)
        enhanced = apply_true_peak_limiter(enhanced, settings.limiter_ceiling_dbfs)

        return PipelineResult(
            frame=enhanced,
            features=features,
            applied_settings=settings,
        )

    def _settings_from_controls(self, controls: EnhancementControls) -> EnhancementSettings:
        return EnhancementSettings(
            target_loudness_dbfs=self.profile.loudness_target_lufs + controls.loudness_boost_db,
            limiter_ceiling_dbfs=self.profile.limiter_ceiling_dbfs,
            clarity_gain_db=self.profile.presence_db + self.profile.air_db * 0.5,
            warmth_gain_db=self.profile.bass_tilt_db,
            stereo_width=min(self.profile.stereo_width, controls.stereo_width),
        )
