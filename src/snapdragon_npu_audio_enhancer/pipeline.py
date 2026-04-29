from __future__ import annotations

from dataclasses import dataclass

from .audio_frame import AudioFrame
from .dsp import DynamicEq, LoudnessNormalizer, StereoWidener, TruePeakLimiter
from .inference import EnhancementFeatures, InferenceConfig, InferenceRouter, Provider
from .profiles import ListeningProfile, ServiceProfile, profile_for_service


@dataclass(frozen=True)
class EnhancementReport:
    service: str
    provider: Provider
    input_rms_dbfs: float
    output_rms_dbfs: float
    input_peak_dbfs: float
    output_peak_dbfs: float
    features: EnhancementFeatures | None

    def to_dict(self) -> dict[str, object]:
        return {
            "service": self.service,
            "provider": self.provider.value,
            "input_rms_dbfs": self.input_rms_dbfs,
            "output_rms_dbfs": self.output_rms_dbfs,
            "input_peak_dbfs": self.input_peak_dbfs,
            "output_peak_dbfs": self.output_peak_dbfs,
            "features": None
            if self.features is None
            else {
                "clarity": self.features.clarity,
                "warmth": self.features.warmth,
                "transient": self.features.transient,
                "stereo_width": self.features.stereo_width,
                "confidence": self.features.confidence,
            },
        }


class EnhancementPipeline:
    """Service-agnostic PCM enhancer intended to sit after WASAPI loopback capture."""

    def __init__(
        self,
        profile: ServiceProfile,
        listening_profile: ListeningProfile | None = None,
        frame_duration_ms: float = 10.0,
        enable_neural: bool = True,
        inference_config: InferenceConfig | None = None,
    ) -> None:
        self.profile = profile
        self.listening_profile = listening_profile or ListeningProfile()
        self.frame_duration_ms = frame_duration_ms
        self.enable_neural = enable_neural
        self.router = InferenceRouter(inference_config or InferenceConfig(prefer_npu=enable_neural))
        self.provider = self.router.select_provider()

    @classmethod
    def for_service(
        cls,
        service_name: str,
        listening_profile: ListeningProfile | None = None,
    ) -> "EnhancementPipeline":
        return cls(profile_for_service(service_name), listening_profile)

    def process_frame(self, frame: AudioFrame) -> AudioFrame:
        return self.process(frame)[0]

    def process(self, frame: AudioFrame) -> tuple[AudioFrame, EnhancementReport]:
        features = self.router.estimate_features(frame) if self.enable_neural else None
        profile = self._profile_with_listener_preferences(features)

        enhanced = LoudnessNormalizer(
            target_lufs=profile.loudness_target_lufs,
            max_gain_db=profile.max_gain_db,
        ).process(frame)
        enhanced = DynamicEq(
            low_gain_db=profile.low_shelf_db,
            presence_gain_db=profile.presence_db,
            air_gain_db=profile.air_db,
            sample_rate=frame.sample_rate,
        ).process(enhanced)
        enhanced = StereoWidener(width=profile.stereo_width).process(enhanced)
        enhanced = TruePeakLimiter(ceiling_dbfs=profile.limiter_ceiling_dbfs).process(enhanced)

        return enhanced, EnhancementReport(
            service=str(profile.service),
            provider=self.provider,
            input_rms_dbfs=frame.rms_dbfs,
            output_rms_dbfs=enhanced.rms_dbfs,
            input_peak_dbfs=frame.peak_dbfs,
            output_peak_dbfs=enhanced.peak_dbfs,
            features=features,
        )

    def _profile_with_listener_preferences(
        self, features: EnhancementFeatures | None
    ) -> ServiceProfile:
        service = self.profile
        listener = self.listening_profile

        bass = service.low_shelf_db + listener.bass_preference_db
        mids = service.presence_db + listener.vocal_clarity_preference_db
        treble = service.air_db + listener.treble_preference_db

        if listener.low_volume_mode:
            bass += 0.9
            treble += 0.7

        if features and features.confidence > 0.2:
            bass += (features.warmth - 0.5) * 0.8
            mids += (features.clarity - 0.5) * 0.6
            treble += (features.clarity - features.warmth) * 0.4

        target_lufs = service.loudness_target_lufs + listener.loudness_offset_db
        target_lufs = max(-20.0, min(-12.0, target_lufs))

        return ServiceProfile(
            service=service.service,
            loudness_target_lufs=target_lufs,
            low_shelf_db=max(-4.0, min(4.0, bass)),
            presence_db=max(-3.0, min(3.0, mids)),
            air_db=max(-4.0, min(4.0, treble)),
            stereo_width=min(service.stereo_width, listener.max_stereo_width),
            max_gain_db=service.max_gain_db,
            limiter_ceiling_dbfs=service.limiter_ceiling_dbfs,
        )
