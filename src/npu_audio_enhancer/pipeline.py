from __future__ import annotations

from dataclasses import dataclass

from .audio import AudioBuffer, ensure_stereo_float32
from .dsp import EnhancementMetrics, DspEnhancer
from .inference import AudioFeatures, NpuFeatureExtractor
from .profiles import EnhancementProfile, service_profile


@dataclass(frozen=True)
class EnhancementSettings:
    service: str = "spotify"
    headphone_profile: str = "generic"
    prefer_npu: bool = True


@dataclass(frozen=True)
class PipelineResult:
    audio: AudioBuffer
    features: AudioFeatures
    metrics: EnhancementMetrics
    profile: EnhancementProfile


class EnhancementPipeline:
    """Coordinates capture-format normalization, NPU inference, and DSP post-processing."""

    def __init__(
        self,
        settings: EnhancementSettings | None = None,
        feature_extractor: NpuFeatureExtractor | None = None,
    ) -> None:
        self.settings = settings or EnhancementSettings()
        self.profile = service_profile(self.settings.service, self.settings.headphone_profile)
        self.feature_extractor = feature_extractor or NpuFeatureExtractor(prefer_npu=self.settings.prefer_npu)
        self.dsp = DspEnhancer(self.profile)

    def process(self, audio: AudioBuffer) -> PipelineResult:
        normalized = ensure_stereo_float32(audio)
        features = self.feature_extractor.extract(normalized)
        enhanced, metrics = self.dsp.process(normalized, features.as_dict())
        return PipelineResult(
            audio=enhanced,
            features=features,
            metrics=metrics,
            profile=self.profile,
        )
