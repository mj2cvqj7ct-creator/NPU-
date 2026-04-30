"""Streaming enhancement pipeline for captured music app PCM audio."""

from __future__ import annotations

from dataclasses import dataclass, field

from .audio_frame import AudioFrame
from .dsp import (
    DynamicEqualizer,
    EnhancementControls,
    LoudnessNormalizer,
    StereoWidener,
    TransientProtector,
    TruePeakLimiter,
    db_to_linear,
)
from .inference import InferenceBackend, InferenceConfig, MusicFeatures, select_backend
from .service_profiles import MusicService, ServiceProfile, get_service_profile


@dataclass(frozen=True)
class EnhancementConfig:
    """User-visible configuration for the real-time enhancer."""

    service: MusicService | str = MusicService.GENERIC
    intensity: float = 0.75
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def bounded_intensity(self) -> float:
        return max(0.0, min(1.0, self.intensity))


@dataclass(frozen=True)
class EnhancementResult:
    """A processed audio block plus metadata useful for telemetry/UI."""

    frame: AudioFrame
    backend_kind: str
    service: MusicService
    features: MusicFeatures
    controls: EnhancementControls
    loudness_gain_db: float


class EnhancementPipeline:
    """Low-latency DSP + NPU inference chain for 10-20 ms stereo frames."""

    def __init__(
        self,
        config: EnhancementConfig | None = None,
        backend: InferenceBackend | None = None,
    ) -> None:
        self.config = config or EnhancementConfig()
        self.profile = get_service_profile(self.config.service)
        self.backend = backend or select_backend(self.config.inference)
        self._loudness = LoudnessNormalizer(target_dbfs=self.profile.target_lufs)
        self._eq = DynamicEqualizer()
        self._transients = TransientProtector()
        self._widener = StereoWidener()
        self._limiter = TruePeakLimiter()

    def process(self, frame: AudioFrame) -> EnhancementResult:
        """Enhance one short stereo frame and return bounded output."""

        normalized, loudness = self._loudness.apply(frame)
        features = self.backend.extract_features(normalized)
        controls = self._controls_from_features(features, self.profile)
        neural = self.backend.enhance(normalized, features)
        shaped = self._eq.process(neural, controls, service_tilt=1.0 + self.config.bounded_intensity() * 0.35)
        shaped = self._transients.process(shaped, controls)
        shaped = self._widener.process(shaped, controls)
        limited = self._limiter.process(shaped)
        return EnhancementResult(
            frame=limited,
            backend_kind=self.backend.kind.value,
            service=self.profile.service,
            features=features,
            controls=controls,
            loudness_gain_db=loudness.gain_db,
        )

    def _controls_from_features(
        self,
        features: MusicFeatures,
        profile: ServiceProfile,
    ) -> EnhancementControls:
        intensity = self.config.bounded_intensity()
        return EnhancementControls(
            vocal_clarity=_scaled(profile.presence_db / 2.0, features.vocal_presence, intensity),
            bass_weight=_scaled(profile.low_shelf_db / 2.0, 1.0 - features.bass_weight * 0.35, intensity),
            transient_restore=_scaled(profile.transient_restore, features.transient_softness, intensity),
            stereo_width=_scaled(profile.stereo_width - 1.0, 12.0, intensity),
            air=_scaled(profile.air_db / 2.0, 1.0 - features.brightness * 0.5, intensity),
        )


class StreamingEnhancer:
    """Accumulates arbitrary PCM samples into fixed-size processing frames."""

    def __init__(self, pipeline: EnhancementPipeline, frame_size: int = 960) -> None:
        if frame_size <= 0:
            raise ValueError("frame_size must be positive")
        self.pipeline = pipeline
        self.frame_size = frame_size
        self._pending: list[tuple[float, float]] = []

    def push(self, frame: AudioFrame) -> list[EnhancementResult]:
        """Push any sized block and receive zero or more processed frames."""

        self._pending.extend(frame.samples)
        results: list[EnhancementResult] = []
        while len(self._pending) >= self.frame_size:
            chunk = self._pending[: self.frame_size]
            del self._pending[: self.frame_size]
            results.append(self.pipeline.process(AudioFrame(frame.sample_rate, tuple(chunk))))
        return results

    def flush(self, sample_rate: int) -> EnhancementResult | None:
        """Process a final partial frame, if present."""

        if not self._pending:
            return None
        chunk = tuple(self._pending)
        self._pending.clear()
        return self.pipeline.process(AudioFrame(sample_rate, chunk))


def _scaled(profile_value: float, feature_value: float, intensity: float) -> float:
    return max(0.0, min(1.0, profile_value * feature_value * intensity))


__all__ = [
    "EnhancementConfig",
    "EnhancementPipeline",
    "EnhancementResult",
    "StreamingEnhancer",
    "db_to_linear",
]
