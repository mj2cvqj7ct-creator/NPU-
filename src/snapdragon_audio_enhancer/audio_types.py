"""Shared audio data structures for the enhancement pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import math


Sample = float
StereoFrame = tuple[Sample, Sample]


@dataclass(frozen=True)
class AudioFeatures:
    """Compact features passed to the NPU or deterministic CPU fallback."""

    peak: float
    rms: float
    crest_factor: float
    low_band_energy: float
    high_band_energy: float
    stereo_correlation: float


@dataclass(frozen=True)
class EnhancementTelemetry:
    """Metrics collected while processing one buffer."""

    backend: str
    service: str
    sample_rate: int
    input_peak: float
    output_peak: float
    input_rms: float
    output_rms: float
    gain_db: float
    limiter_reductions: int
    used_npu: bool


@dataclass(frozen=True)
class AudioBuffer:
    """Stereo PCM block represented as normalized float samples."""

    sample_rate: int
    frames: tuple[StereoFrame, ...]

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

    @property
    def channels(self) -> int:
        return 2

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.sample_rate

    @property
    def peak(self) -> float:
        return max((max(abs(left), abs(right)) for left, right in self.frames), default=0.0)

    @property
    def rms(self) -> float:
        if not self.frames:
            return 0.0
        total = sum(left * left + right * right for left, right in self.frames)
        return math.sqrt(total / (self.frame_count * self.channels))

    def with_frames(self, frames: list[StereoFrame] | tuple[StereoFrame, ...]) -> AudioBuffer:
        return AudioBuffer(sample_rate=self.sample_rate, frames=tuple(frames))

    def features(self) -> AudioFeatures:
        if not self.frames:
            return AudioFeatures(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        peak = self.peak
        rms = self.rms
        low_energy = 0.0
        high_energy = 0.0
        previous_mid = 0.0
        left_energy = 0.0
        right_energy = 0.0
        cross_energy = 0.0

        for left, right in self.frames:
            mid = (left + right) * 0.5
            low_energy += mid * mid
            high = mid - previous_mid
            high_energy += high * high
            previous_mid = mid
            left_energy += left * left
            right_energy += right * right
            cross_energy += left * right

        normalizer = max(low_energy + high_energy, 1e-12)
        denominator = math.sqrt(max(left_energy * right_energy, 1e-12))
        correlation = max(-1.0, min(1.0, cross_energy / denominator))
        return AudioFeatures(
            peak=peak,
            rms=rms,
            crest_factor=peak / max(rms, 1e-12),
            low_band_energy=low_energy / normalizer,
            high_band_energy=high_energy / normalizer,
            stereo_correlation=correlation,
        )
