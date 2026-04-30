from __future__ import annotations

from dataclasses import dataclass

from .audio_types import AudioFrame
from .dsp import EnhancementPipeline as DspPipeline
from .dsp import FrameMetrics
from .inference import EnhancementControls, ProviderSelection, select_provider
from .profile import EnhancementProfile, MusicService


@dataclass(frozen=True)
class EnhancementConfig:
    service: MusicService = MusicService.GENERIC
    frame_milliseconds: int = 20
    prefer_npu: bool = True

    def frame_size(self, sample_rate: int) -> int:
        if self.frame_milliseconds <= 0:
            raise ValueError("frame_milliseconds must be positive")
        return max(1, sample_rate * self.frame_milliseconds // 1000)


@dataclass(frozen=True)
class ProcessingReport:
    provider_name: str
    provider_reason: str
    input_peak: float
    output_peak: float
    input_rms: float
    output_rms: float
    frames_processed: int


class EnhancementPipeline:
    """Chunked audio enhancement pipeline with NPU-ready inference controls."""

    def __init__(self, config: EnhancementConfig) -> None:
        self.config = config
        self.profile = EnhancementProfile.for_service(config.service)
        self.provider_selection: ProviderSelection = select_provider(prefer_npu=config.prefer_npu)

    def process(self, audio: AudioFrame) -> tuple[AudioFrame, ProcessingReport]:
        dsp = DspPipeline(profile=self.profile, sample_rate=audio.sample_rate)
        frame_size = self.config.frame_size(audio.sample_rate)
        processed_frames: list[tuple[float, float]] = []
        last_metrics = FrameMetrics(rms=0.0, peak=0.0, low_energy=0.0, mid_energy=0.0, high_energy=0.0)

        for chunk in audio.chunks(frame_size):
            controls = self.provider_selection.provider.analyze(chunk).clamp()
            enhanced, last_metrics = dsp.process(chunk, controls)
            processed_frames.extend(enhanced.frames)

        output = AudioFrame(sample_rate=audio.sample_rate, frames=tuple(processed_frames), channels=audio.channels)
        report = ProcessingReport(
            provider_name=self.provider_selection.provider.name,
            provider_reason=self.provider_selection.reason,
            input_peak=audio.peak(),
            output_peak=output.peak(),
            input_rms=audio.rms(),
            output_rms=output.rms(),
            frames_processed=audio.frame_count,
        )
        return output, report
