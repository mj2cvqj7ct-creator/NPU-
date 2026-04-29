"""Local-only enhancement profiles for supported music services."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class MusicService(StrEnum):
    """Streaming services handled through OS audio output."""

    SPOTIFY = "spotify"
    APPLE_MUSIC = "apple_music"
    YOUTUBE_MUSIC = "youtube_music"
    GENERIC = "generic"


@dataclass(frozen=True)
class EnhancementProfile:
    """Tuning values for the rule-based audio pipeline.

    Values are intentionally conservative. The goal is a stable baseline that can
    be refined by local preference learning and NPU-backed feature inference.
    """

    service: MusicService
    loudness_target_lufs: float
    low_shelf_db: float
    presence_db: float
    air_db: float
    stereo_width: float
    max_gain_db: float = 6.0
    limiter_ceiling_dbfs: float = -1.0


ServiceProfile = EnhancementProfile
ServiceName = MusicService


@dataclass(frozen=True)
class ListeningProfile:
    """Local preference controls learned from user choices, never sent to services."""

    bass_preference_db: float = 0.0
    vocal_clarity_preference_db: float = 0.0
    treble_preference_db: float = 0.0
    loudness_offset_db: float = 0.0
    max_stereo_width: float = 1.08
    low_volume_mode: bool = False


PROFILES: dict[MusicService, EnhancementProfile] = {
    MusicService.SPOTIFY: EnhancementProfile(
        service=MusicService.SPOTIFY,
        loudness_target_lufs=-15.0,
        low_shelf_db=1.2,
        presence_db=0.8,
        air_db=0.6,
        stereo_width=1.04,
    ),
    MusicService.APPLE_MUSIC: EnhancementProfile(
        service=MusicService.APPLE_MUSIC,
        loudness_target_lufs=-16.0,
        low_shelf_db=0.8,
        presence_db=0.7,
        air_db=0.9,
        stereo_width=1.02,
    ),
    MusicService.YOUTUBE_MUSIC: EnhancementProfile(
        service=MusicService.YOUTUBE_MUSIC,
        loudness_target_lufs=-14.0,
        low_shelf_db=1.0,
        presence_db=1.1,
        air_db=0.7,
        stereo_width=1.03,
    ),
    MusicService.GENERIC: EnhancementProfile(
        service=MusicService.GENERIC,
        loudness_target_lufs=-15.0,
        low_shelf_db=0.8,
        presence_db=0.8,
        air_db=0.6,
        stereo_width=1.02,
    ),
}


def get_profile(service: MusicService | str | None) -> EnhancementProfile:
    """Return a supported profile, defaulting to a generic system profile."""

    if service is None:
        return PROFILES[MusicService.GENERIC]

    try:
        service_id = MusicService(str(service).lower().replace("-", "_"))
    except ValueError:
        service_id = MusicService.GENERIC

    return PROFILES[service_id]


def profile_for_service(service: MusicService | str | None) -> EnhancementProfile:
    return get_profile(service)
