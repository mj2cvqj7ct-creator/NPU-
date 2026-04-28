from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .dsp import EnhancementDecision, FeatureExtractor, RuleBasedEnhancer, SERVICE_PROFILES, StereoFrame
from .inference import BackendInfo, BackendKind, NpuEnhancer


class ServiceProfile(str, Enum):
    SPOTIFY = "spotify"
    APPLE_MUSIC = "apple_music"
    YOUTUBE_MUSIC = "youtube_music"
    GENERIC = "generic"


@dataclass(frozen=True)
class EnhancerConfig:
    service: ServiceProfile = ServiceProfile.GENERIC
    sample_rate: int = 48_000
    frame_size: int = 960
    preferred_backend: BackendKind | None = None


@dataclass(frozen=True)
class ProcessingReport:
    service: ServiceProfile
    backend: BackendInfo
    input_peak: float
    output_peak: float
    decision: EnhancementDecision


class AudioEnhancer:
    """Frame-based PCM post-processing pipeline for service-agnostic audio."""

    def __init__(self, config: EnhancerConfig | None = None) -> None:
        self.config = config or EnhancerConfig()
        if self.config.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.config.frame_size <= 0:
            raise ValueError("frame_size must be positive")
        self._features = FeatureExtractor()
        self._dsp = RuleBasedEnhancer()
        self._npu = NpuEnhancer(preferred=self.config.preferred_backend)

    @property
    def inference_backend(self) -> BackendInfo:
        return self._npu.backend

    def process_frame(self, frame: list[StereoFrame]) -> tuple[list[StereoFrame], ProcessingReport]:
        profile = SERVICE_PROFILES[self.config.service.value]
        input_features = self._features.analyze(frame)
        model = self._npu.infer(frame, input_features)
        base_decision = self._dsp.decide(input_features, profile)
        decision = EnhancementDecision(
            gain=base_decision.gain,
            bass_tilt=min(0.06, base_decision.bass_tilt + model.warmth * 0.08),
            presence_tilt=min(0.08, base_decision.presence_tilt + model.clarity * 0.10),
            stereo_width=min(1.08, base_decision.stereo_width + model.spatial * 0.08),
        )
        processed = self._dsp.process(frame, profile, decision)
        output_features = self._features.analyze(processed)
        report = ProcessingReport(
            service=self.config.service,
            backend=self.inference_backend,
            input_peak=input_features.peak,
            output_peak=output_features.peak,
            decision=decision,
        )
        return processed, report

    def process_stream(self, frames: list[StereoFrame]) -> list[StereoFrame]:
        output: list[StereoFrame] = []
        for offset in range(0, len(frames), self.config.frame_size):
            processed, _ = self.process_frame(frames[offset : offset + self.config.frame_size])
            output.extend(processed)
        return output


def create_enhancer(config: EnhancerConfig | None = None) -> AudioEnhancer:
    return AudioEnhancer(config)
