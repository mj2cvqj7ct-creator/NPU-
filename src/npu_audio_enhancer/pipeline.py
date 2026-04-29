from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .dsp import (
    AudioFrame,
    FrameStats,
    apply_dynamic_eq,
    db_to_linear,
    frame_stats,
    normalize_loudness,
    soft_knee_limiter,
)
from .profiles import ServiceProfile, get_service_profile


class NpuEnhancer(Protocol):
    """Boundary for Snapdragon X NPU inference backends."""

    def enhance(self, frame: AudioFrame, profile: ServiceProfile) -> AudioFrame:
        """Return an enhanced frame with the same channel count and length."""


@dataclass(frozen=True)
class PassthroughNpuEnhancer:
    """Safe default until QNN/ONNX Runtime EP integration is available."""

    def enhance(self, frame: AudioFrame, profile: ServiceProfile) -> AudioFrame:
        return [channel.copy() for channel in frame]


@dataclass(frozen=True)
class EnhancementResult:
    frame: AudioFrame
    before: FrameStats
    after: FrameStats
    service: str
    npu_backend: str


@dataclass
class EnhancementPipeline:
    """Low-latency PCM enhancement chain for music service output."""

    service_name: str
    npu: NpuEnhancer = PassthroughNpuEnhancer()
    sample_rate: int = 48_000

    def __post_init__(self) -> None:
        if self.sample_rate != 48_000:
            raise ValueError("the first implementation expects 48 kHz PCM frames")
        self.profile = get_service_profile(self.service_name)

    def process(self, frame: AudioFrame) -> EnhancementResult:
        normalized, normalization_stats = normalize_loudness(
            frame,
            target_lufs=self.profile.target_lufs,
            max_gain_db=self.profile.max_gain_db,
        )

        equalized = apply_dynamic_eq(normalized, self.profile.eq)
        enhanced = self.npu.enhance(equalized, self.profile)
        limited = soft_knee_limiter(
            enhanced,
            threshold=db_to_linear(self.profile.limiter_ceiling_db),
        )
        after = frame_stats(limited)
        return EnhancementResult(
            frame=limited,
            before=normalization_stats.input,
            after=after,
            service=self.profile.service_name,
            npu_backend=type(self.npu).__name__,
        )
