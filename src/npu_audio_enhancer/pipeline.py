from __future__ import annotations

from dataclasses import dataclass

from .dsp import ProcessingMetrics, analyze_frame, enhance_frame, extract_features
from .frame import AudioFrame
from .inference import EnhancementBackend, select_backend
from .profiles import MusicService, ServiceProfile, resolve_profile


@dataclass(frozen=True)
class PipelineResult:
    frame: AudioFrame
    service: MusicService
    backend: str
    metrics: ProcessingMetrics


class AudioEnhancementPipeline:
    """Low-latency frame processor for system-wide music enhancement."""

    def __init__(
        self,
        service: str = "generic",
        *,
        backend: EnhancementBackend | None = None,
        prefer_npu: bool = True,
        model_path: str | None = None,
    ) -> None:
        self.profile = resolve_profile(service)
        self.backend = backend or select_backend(prefer_npu=prefer_npu, model_path=model_path)

    def process(self, frame: AudioFrame) -> PipelineResult:
        analysis = analyze_frame(frame)
        features = extract_features(frame, analysis)
        inference = self.backend.infer(frame, features, self.profile)
        processed, metrics = enhance_frame(frame, self.profile, inference, analysis)
        return PipelineResult(
            frame=processed,
            service=self.profile.service,
            backend=self.backend.kind.value,
            metrics=metrics,
        )

    def switch_service(self, service: str | MusicService) -> None:
        self.profile = resolve_profile(service)


EnhancementPipeline = AudioEnhancementPipeline
