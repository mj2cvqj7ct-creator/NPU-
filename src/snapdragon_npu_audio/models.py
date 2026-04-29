from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Sequence


StereoSample = tuple[float, float]


class ServiceProfile(str, Enum):
    """Streaming sources the enhancer can tune for without modifying apps."""

    SPOTIFY = "spotify"
    APPLE_MUSIC = "apple_music"
    YOUTUBE_MUSIC = "youtube_music"
    GENERIC = "generic"


@dataclass(frozen=True)
class AudioFrame:
    """A short interleaved stereo PCM block represented as normalized floats."""

    sample_rate_hz: int
    channels: int
    samples: Sequence[StereoSample]

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.channels != 2:
            raise ValueError("the prototype currently supports stereo frames only")

        normalized = tuple((float(left), float(right)) for left, right in self.samples)
        for left, right in normalized:
            if not -4.0 <= left <= 4.0 or not -4.0 <= right <= 4.0:
                raise ValueError("samples must be normalized PCM-like values")
        object.__setattr__(self, "samples", normalized)

    @property
    def peak(self) -> float:
        if not self.samples:
            return 0.0
        return max(max(abs(left), abs(right)) for left, right in self.samples)

    @property
    def rms(self) -> float:
        if not self.samples:
            return 0.0
        square_sum = sum(left * left + right * right for left, right in self.samples)
        return math.sqrt(square_sum / (len(self.samples) * self.channels))

    def map_samples(self, transform: Callable[[float], float]) -> "AudioFrame":
        return self.with_samples(
            (transform(left), transform(right)) for left, right in self.samples
        )

    def with_samples(self, samples: Sequence[StereoSample]) -> "AudioFrame":
        return AudioFrame(
            sample_rate_hz=self.sample_rate_hz,
            channels=self.channels,
            samples=tuple(samples),
        )


@dataclass(frozen=True)
class EnhancementSettings:
    """Safety-bounded tuning for one local playback service profile."""

    target_loudness_lufs: float = -16.0
    max_gain_db: float = 12.0
    bass_tilt_db: float = 0.3
    presence_tilt_db: float = 0.9
    stereo_width: float = 1.02
    limiter_ceiling: float = 0.98

    def __post_init__(self) -> None:
        if self.max_gain_db < 0.0:
            raise ValueError("max_gain_db must be non-negative")
        if not 0.75 <= self.stereo_width <= 1.35:
            raise ValueError("stereo_width must stay within a conservative range")
        if not 0.0 < self.limiter_ceiling <= 1.0:
            raise ValueError("limiter_ceiling must be in (0, 1]")


@dataclass(frozen=True)
class FrameFeatures:
    rms: float
    peak: float
    crest_factor: float
    stereo_correlation: float
    low_band_energy: float
    high_band_energy: float


InferenceFeatures = FrameFeatures


@dataclass(frozen=True)
class EnhancementDecision:
    bass_boost_db: float
    clarity_boost_db: float
    low_volume_compensation_db: float
    transient_restore: float
    backend_name: str
