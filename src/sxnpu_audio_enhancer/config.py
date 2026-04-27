from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np


class MusicService(str, Enum):
    """Known playback sources handled only through post-OS PCM output."""

    SPOTIFY = "spotify"
    APPLE_MUSIC = "apple_music"
    YOUTUBE_MUSIC = "youtube_music"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ServicePolicy:
    service: MusicService
    direct_app_modification: bool = False
    preserve_drm_boundary: bool = True
    allow_recording: bool = False
    notes: tuple[str, ...] = ()


SERVICE_POLICIES: dict[MusicService, ServicePolicy] = {
    MusicService.SPOTIFY: ServicePolicy(
        MusicService.SPOTIFY,
        notes=("Enhance WASAPI loopback PCM only.", "Keep preference learning local."),
    ),
    MusicService.APPLE_MUSIC: ServicePolicy(
        MusicService.APPLE_MUSIC,
        notes=("Post-process rendered lossless or AAC PCM.", "Avoid touching library files."),
    ),
    MusicService.YOUTUBE_MUSIC: ServicePolicy(
        MusicService.YOUTUBE_MUSIC,
        notes=("Handle browser/app PCM uniformly.", "Correct loudness jumps between videos."),
    ),
    MusicService.UNKNOWN: ServicePolicy(
        MusicService.UNKNOWN,
        notes=("Apply conservative system-wide enhancement.",),
    ),
}


def db_to_linear(db: float) -> float:
    return float(10.0 ** (db / 20.0))


@dataclass(frozen=True)
class EnhancerConfig:
    sample_rate: int = 48_000
    channels: int = 2
    frame_ms: float = 10.0
    target_lufs: float = -18.0
    max_gain_db: float = 6.0
    limiter_ceiling_dbfs: float = -1.0
    true_peak_limit_db: float | None = None
    stereo_width: float = 1.05
    low_volume_bass_lift_db: float = 1.5
    headphone_eq_db: tuple[float, float, float] = (0.8, 0.0, 0.6)
    model_path: Path | None = None
    npu_feature_names: tuple[str, ...] = field(
        default=("rms", "peak", "spectral_centroid", "zero_crossing_rate")
    )

    def __post_init__(self) -> None:
        if self.channels != 2:
            raise ValueError("the realtime enhancer currently expects stereo PCM")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.frame_ms <= 0:
            raise ValueError("frame_ms must be positive")

    @property
    def frame_size(self) -> int:
        return max(1, int(self.sample_rate * self.frame_ms / 1000.0))

    @property
    def limiter_ceiling(self) -> float:
        return self.true_peak_linear

    @property
    def true_peak_linear(self) -> float:
        ceiling_db = (
            self.limiter_ceiling_dbfs
            if self.true_peak_limit_db is None
            else self.true_peak_limit_db
        )
        return db_to_linear(ceiling_db)

    @property
    def max_gain_linear(self) -> float:
        return db_to_linear(self.max_gain_db)

    @property
    def target_rms_linear(self) -> float:
        return db_to_linear(self.target_lufs)


EnhancementConfig = EnhancerConfig
