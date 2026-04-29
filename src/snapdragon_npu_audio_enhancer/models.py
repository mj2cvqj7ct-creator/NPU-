from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class StreamingService(str, Enum):
    """Supported service profiles for post-processing OS output audio."""

    GENERIC = "generic"
    SPOTIFY = "spotify"
    APPLE_MUSIC = "apple_music"
    YOUTUBE_MUSIC = "youtube_music"


Frame = list[tuple[float, float]]


@dataclass(frozen=True)
class EnhancementConfig:
    """Safe real-time controls for a single streaming enhancement pass."""

    sample_rate: int = 48_000
    service: StreamingService = StreamingService.GENERIC
    target_lufs: float = -16.0
    true_peak_ceiling: float = 0.8912509381337456  # -1 dBFS
    clarity: float = 0.35
    warmth: float = 0.25
    stereo_width: float = 1.0
    transient_restore: float = 0.15

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not 0.0 < self.true_peak_ceiling <= 1.0:
            raise ValueError("true_peak_ceiling must be within 0.0..1.0")


@dataclass(frozen=True)
class AudioAnalysis:
    """Compact frame features intended for NPU or fallback inference."""

    rms: float
    peak: float
    crest_factor_db: float
    low_energy: float
    mid_energy: float
    high_energy: float
    stereo_correlation: float
    zero_crossing_rate: float


@dataclass(frozen=True)
class EnhancementReport:
    """Diagnostics for tests, logging, and user-facing quality telemetry."""

    sample_rate: int
    service: StreamingService
    input_peak: float
    output_peak: float
    input_rms_db: float
    output_rms_db: float
    limiter_reduction_db: float
    provider: str
    npu_accelerated: bool

    @classmethod
    def empty(cls, config: EnhancementConfig) -> "EnhancementReport":
        return cls(
            sample_rate=config.sample_rate,
            service=config.service,
            input_peak=0.0,
            output_peak=0.0,
            input_rms_db=-240.0,
            output_rms_db=-240.0,
            limiter_reduction_db=0.0,
            provider="rule-based",
            npu_accelerated=False,
        )

