from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from npu_audio_enhancer.dsp.frame import AudioFrame
from npu_audio_enhancer.dsp.pipeline import (
    AudioEnhancementPipeline,
    EnhancementConfig,
    EnhancementReport,
)
from npu_audio_enhancer.inference.backend import SnapdragonNpuBackendSelector
from npu_audio_enhancer.profile.model import ListeningPreference


@dataclass(frozen=True)
class EnhancementSettings:
    service_name: str = "system"
    sample_rate: int = 48_000
    channels: int = 2
    config: EnhancementConfig | None = None


class StreamingEnhancer:
    """Service-agnostic enhancer for OS-level PCM captured from music apps."""

    def __init__(
        self,
        settings: EnhancementSettings | None = None,
        *,
        profile: ListeningPreference | None = None,
        backend_selector: SnapdragonNpuBackendSelector | None = None,
    ) -> None:
        self.settings = settings or EnhancementSettings()
        self.profile = profile or ListeningPreference()
        self.backend_selector = backend_selector or SnapdragonNpuBackendSelector()
        config = self.settings.config or self.profile.to_enhancement_config()
        self.pipeline = AudioEnhancementPipeline(config)

    def process_interleaved(self, samples: Iterable[float]) -> tuple[list[float], EnhancementReport]:
        frame = AudioFrame.from_interleaved(
            samples,
            channels=self.settings.channels,
            sample_rate=self.settings.sample_rate,
        )
        inference = self.backend_selector.infer(frame, self.settings.service_name)
        processed, report = self.pipeline.process(
            frame,
            service_profile=self.settings.service_name,
            npu_backend=inference.backend_name,
            neural_gain=inference.neural_gain,
        )
        return processed.to_interleaved(), report
