"""Service and playback profiles for the audio enhancement pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class MusicService(StrEnum):
    """Supported music service labels for local OS-output processing."""

    SPOTIFY = "spotify"
    APPLE_MUSIC = "apple_music"
    YOUTUBE_MUSIC = "youtube_music"
    GENERIC = "generic"


@dataclass(frozen=True)
class EnhancementProfile:
    """Per-service tuning applied after PCM audio reaches the OS mixer."""

    service: MusicService
    loudness_target_lufs: float
    bass_tilt_db: float
    presence_db: float
    air_db: float
    stereo_width: float
    transient_restore: float
    limiter_ceiling_dbfs: float = -1.0


ServiceProfile = EnhancementProfile


_PROFILES: dict[MusicService, EnhancementProfile] = {
    MusicService.SPOTIFY: EnhancementProfile(
        service=MusicService.SPOTIFY,
        loudness_target_lufs=-15.0,
        bass_tilt_db=0.8,
        presence_db=1.2,
        air_db=0.8,
        stereo_width=1.04,
        transient_restore=0.18,
    ),
    MusicService.APPLE_MUSIC: EnhancementProfile(
        service=MusicService.APPLE_MUSIC,
        loudness_target_lufs=-16.0,
        bass_tilt_db=0.4,
        presence_db=0.8,
        air_db=1.0,
        stereo_width=1.02,
        transient_restore=0.12,
    ),
    MusicService.YOUTUBE_MUSIC: EnhancementProfile(
        service=MusicService.YOUTUBE_MUSIC,
        loudness_target_lufs=-14.0,
        bass_tilt_db=0.7,
        presence_db=1.0,
        air_db=0.6,
        stereo_width=1.01,
        transient_restore=0.16,
    ),
    MusicService.GENERIC: EnhancementProfile(
        service=MusicService.GENERIC,
        loudness_target_lufs=-15.5,
        bass_tilt_db=0.5,
        presence_db=0.7,
        air_db=0.5,
        stereo_width=1.0,
        transient_restore=0.1,
    ),
}


SERVICE_PROFILES = _PROFILES.copy()


def get_profile(service: MusicService | str) -> EnhancementProfile:
    """Return a conservative profile for a supported service."""

    try:
        normalized = MusicService(service)
    except ValueError:
        normalized = MusicService.GENERIC
    return _PROFILES[normalized]


get_service_profile = get_profile
