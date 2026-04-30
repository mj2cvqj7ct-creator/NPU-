from __future__ import annotations

from dataclasses import dataclass, field

from .dsp import AudioMetrics, EnhancementSettings, analyze_frame, enhance_frame
from .inference import InferenceProvider, InferenceTuning, create_inference_provider
from .profiles import ServiceProfile, get_service_profile


@dataclass(frozen=True)
class EnhancementResult:
    """Processed frame plus the metrics that drove the enhancement."""

    samples: list[list[float]]
    metrics: AudioMetrics
    service: ServiceProfile
    provider: str
    tuning: InferenceTuning
    metadata: dict[str, str]

    @property
    def frames(self) -> list[list[float]]:
        """Compatibility alias for offline file processing."""

        return self.samples


@dataclass
class AudioEnhancementPipeline:
    """Small real-time friendly orchestrator for streamed PCM frames."""

    service_name: str = "generic"
    sample_rate: int = 48_000
    provider_name: str = "auto"
    service: ServiceProfile = field(init=False)
    provider: InferenceProvider = field(init=False)

    def __post_init__(self) -> None:
        self.service = get_service_profile(self.service_name)
        self.provider = create_inference_provider(self.provider_name)

    def process_frame(self, samples: list[list[float]]) -> EnhancementResult:
        input_metrics = analyze_frame(samples)
        tuning = self.provider.infer(input_metrics, self.sample_rate, self.service)
        settings = EnhancementSettings.from_profile(self.service, tuning)
        enhanced, metrics = enhance_frame(samples, self.sample_rate, settings)
        return EnhancementResult(
            samples=enhanced,
            metrics=metrics,
            service=self.service,
            provider=self.provider.name,
            tuning=tuning,
            metadata={
                "service": self.service.display_name,
                "inference_provider": self.provider.name,
                "provider_reason": tuning.provider.reason,
            },
        )

    @classmethod
    def for_service(
        cls,
        service_name: str,
        *,
        sample_rate_hz: int = 48_000,
        preferred_provider: str | None = None,
    ) -> "AudioEnhancementPipeline":
        return cls(
            service_name=service_name,
            sample_rate=sample_rate_hz,
            provider_name=preferred_provider or "auto",
        )

    def process(self, samples: list[list[float]]) -> EnhancementResult:
        return self.process_frame(samples)
