from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


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


@dataclass(frozen=True)
class EnhancementConfig:
    sample_rate: int = 48_000
    channels: int = 2
    frame_ms: float = 10.0
    target_rms_dbfs: float = -18.0
    max_gain_db: float = 6.0
    limiter_ceiling_dbfs: float = -1.0
    stereo_width: float = 1.05
    low_volume_bass_lift_db: float = 1.5
    headphone_eq_db: tuple[float, float, float] = (0.8, 0.0, 0.6)
    npu_feature_names: tuple[str, ...] = field(
        default=("rms", "peak", "spectral_centroid", "zero_crossing_rate")
    )

    @property
    def frame_size(self) -> int:
        return max(1, int(self.sample_rate * self.frame_ms / 1000.0))
