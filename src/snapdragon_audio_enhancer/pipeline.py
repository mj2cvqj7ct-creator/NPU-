"""Block-based enhancement pipeline shared by CLI and future capture code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .dsp import AudioStats, SampleFrame, chunk_frames, enhance_frames, measure
from .inference import EnhancementHints, NpuEnhancementModel, select_backend
from .profiles import ServiceProfile, get_service_profile


@dataclass(frozen=True)
class ProcessReport:
    service: ServiceProfile
    backend: str
    backend_reason: str
    input_stats: AudioStats
    output_stats: AudioStats
    average_detail_gain_db: float
    average_model_confidence: float


@dataclass(frozen=True)
class ProcessResult:
    frames: list[SampleFrame]
    report: ProcessReport


class EnhancementPipeline:
    """Run service-aware DSP with an NPU-compatible inference facade."""

    def __init__(
        self,
        service: str | None = None,
        sample_rate: int = 48_000,
        block_ms: int = 20,
        block_size: int | None = None,
        model: NpuEnhancementModel | None = None,
    ) -> None:
        self.service = get_service_profile(service)
        self.sample_rate = sample_rate
        self.block_size = block_size or max(1, sample_rate * block_ms // 1000)
        self.model = model or NpuEnhancementModel(select_backend())

    def process(self, frames: Sequence[SampleFrame]) -> ProcessResult:
        input_stats = measure(frames)
        output: list[SampleFrame] = []
        hints: list[EnhancementHints] = []

        for block in chunk_frames(frames, self.block_size):
            hint = self.model.infer(block)
            hints.append(hint)
            output.extend(
                enhance_frames(
                    block,
                    self.service.dsp_profile,
                    npu_detail_gain=hint.detail_gain_db,
                )
            )

        output_stats = measure(output)
        hint_count = max(len(hints), 1)
        report = ProcessReport(
            service=self.service,
            backend=self.model.decision.backend.value,
            backend_reason=self.model.decision.reason,
            input_stats=input_stats,
            output_stats=output_stats,
            average_detail_gain_db=sum(h.detail_gain_db for h in hints) / hint_count,
            average_model_confidence=sum(h.confidence for h in hints) / hint_count,
        )
        return ProcessResult(frames=output, report=report)
