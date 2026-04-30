"""Service-aware defaults for post-processing streamed music output."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class MusicService(StrEnum):
    """Known music services that can be tuned after OS-level PCM capture."""

    SPOTIFY = "spotify"
    APPLE_MUSIC = "apple_music"
    YOUTUBE_MUSIC = "youtube_music"
    GENERIC = "generic"


@dataclass(frozen=True)
class ServiceProfile:
    """Conservative DSP defaults for a service's typical output behavior."""

    service: MusicService
    target_lufs: float
    low_shelf_db: float
    presence_db: float
    air_db: float
    transient_restore: float
    stereo_width: float


_PROFILES: dict[MusicService, ServiceProfile] = {
    MusicService.SPOTIFY: ServiceProfile(
        service=MusicService.SPOTIFY,
        target_lufs=-15.0,
        low_shelf_db=1.2,
        presence_db=1.4,
        air_db=0.8,
        transient_restore=0.35,
        stereo_width=1.04,
    ),
    MusicService.APPLE_MUSIC: ServiceProfile(
        service=MusicService.APPLE_MUSIC,
        target_lufs=-16.0,
        low_shelf_db=0.7,
        presence_db=0.9,
        air_db=1.1,
        transient_restore=0.25,
        stereo_width=1.02,
    ),
    MusicService.YOUTUBE_MUSIC: ServiceProfile(
        service=MusicService.YOUTUBE_MUSIC,
        target_lufs=-14.5,
        low_shelf_db=1.0,
        presence_db=1.7,
        air_db=0.7,
        transient_restore=0.45,
        stereo_width=1.03,
    ),
    MusicService.GENERIC: ServiceProfile(
        service=MusicService.GENERIC,
        target_lufs=-15.5,
        low_shelf_db=0.8,
        presence_db=1.0,
        air_db=0.8,
        transient_restore=0.30,
        stereo_width=1.02,
    ),
}


def get_service_profile(service: MusicService | str) -> ServiceProfile:
    """Return DSP defaults for a music service, falling back to generic."""

    try:
        normalized = MusicService(service)
    except ValueError:
        normalized = MusicService.GENERIC
    return _PROFILES[normalized]
