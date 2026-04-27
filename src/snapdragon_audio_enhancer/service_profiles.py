from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MusicService(str, Enum):
    """Supported playback sources for service-aware tuning."""

    SPOTIFY = "spotify"
    APPLE_MUSIC = "apple_music"
    YOUTUBE_MUSIC = "youtube_music"


@dataclass(frozen=True)
class ServiceProfile:
    service: MusicService
    target_loudness_lufs: float
    bass_gain_db: float
    presence_gain_db: float
    air_gain_db: float
    stereo_width: float
    transient_restore: float
    limiter_ceiling: float = 0.98

    @property
    def name(self) -> str:
        return self.service.value


PROFILES: dict[MusicService, ServiceProfile] = {
    MusicService.SPOTIFY: ServiceProfile(
        service=MusicService.SPOTIFY,
        target_loudness_lufs=-15.0,
        bass_gain_db=1.1,
        presence_gain_db=1.4,
        air_gain_db=0.8,
        stereo_width=1.04,
        transient_restore=0.18,
    ),
    MusicService.APPLE_MUSIC: ServiceProfile(
        service=MusicService.APPLE_MUSIC,
        target_loudness_lufs=-16.0,
        bass_gain_db=0.7,
        presence_gain_db=0.9,
        air_gain_db=1.0,
        stereo_width=1.02,
        transient_restore=0.12,
    ),
    MusicService.YOUTUBE_MUSIC: ServiceProfile(
        service=MusicService.YOUTUBE_MUSIC,
        target_loudness_lufs=-14.0,
        bass_gain_db=0.9,
        presence_gain_db=1.7,
        air_gain_db=0.5,
        stereo_width=1.03,
        transient_restore=0.2,
    ),
}


def get_profile(service: MusicService | str) -> ServiceProfile:
    if isinstance(service, str):
        normalized = service.strip().lower().replace("-", "_")
        service = MusicService(normalized)
    return PROFILES[service]


get_service_profile = get_profile
ServiceName = MusicService
