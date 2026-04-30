from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class MusicService(StrEnum):
    SPOTIFY = "spotify"
    APPLE_MUSIC = "apple_music"
    YOUTUBE_MUSIC = "youtube_music"
    GENERIC = "generic"


@dataclass(frozen=True)
class ServiceProfile:
    service: MusicService
    target_lufs: float
    preamp_db: float
    low_shelf_db: float
    presence_db: float
    air_db: float
    stereo_width: float
    transient_restore: float
    limiter_ceiling_db: float = -1.0

    @property
    def gain_linear(self) -> float:
        return 10 ** (self.preamp_db / 20.0)


SERVICE_PROFILES: dict[MusicService, ServiceProfile] = {
    MusicService.SPOTIFY: ServiceProfile(
        service=MusicService.SPOTIFY,
        target_lufs=-15.0,
        preamp_db=-0.5,
        low_shelf_db=0.9,
        presence_db=0.7,
        air_db=0.5,
        stereo_width=1.04,
        transient_restore=0.18,
    ),
    MusicService.APPLE_MUSIC: ServiceProfile(
        service=MusicService.APPLE_MUSIC,
        target_lufs=-16.0,
        preamp_db=-1.0,
        low_shelf_db=0.4,
        presence_db=0.45,
        air_db=0.8,
        stereo_width=1.02,
        transient_restore=0.12,
    ),
    MusicService.YOUTUBE_MUSIC: ServiceProfile(
        service=MusicService.YOUTUBE_MUSIC,
        target_lufs=-14.0,
        preamp_db=-1.2,
        low_shelf_db=0.7,
        presence_db=0.9,
        air_db=0.35,
        stereo_width=1.0,
        transient_restore=0.1,
    ),
    MusicService.GENERIC: ServiceProfile(
        service=MusicService.GENERIC,
        target_lufs=-15.0,
        preamp_db=-0.8,
        low_shelf_db=0.6,
        presence_db=0.5,
        air_db=0.45,
        stereo_width=1.01,
        transient_restore=0.1,
    ),
}


def resolve_profile(service: MusicService | str) -> ServiceProfile:
    if isinstance(service, str):
        normalized_value = service.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "apple": MusicService.APPLE_MUSIC,
            "itunes": MusicService.APPLE_MUSIC,
            "youtube": MusicService.YOUTUBE_MUSIC,
            "yt_music": MusicService.YOUTUBE_MUSIC,
            "ytmusic": MusicService.YOUTUBE_MUSIC,
        }
        if normalized_value in aliases:
            return SERVICE_PROFILES[aliases[normalized_value]]
        service = normalized_value

    try:
        normalized = MusicService(service)
    except ValueError:
        normalized = MusicService.GENERIC
    return SERVICE_PROFILES[normalized]
