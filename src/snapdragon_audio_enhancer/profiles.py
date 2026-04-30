"""Service and listener profiles for local audio enhancement.

The service profiles model the *output characteristics* we can legally adjust
after audio reaches the OS mixer. They do not modify Spotify, Apple Music, or
YouTube Music internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ServiceName(str, Enum):
    """Supported playback service presets."""

    SPOTIFY = "spotify"
    APPLE_MUSIC = "apple_music"
    YOUTUBE_MUSIC = "youtube_music"
    GENERIC = "generic"


@dataclass(frozen=True)
class ServiceProfile:
    """Tuning knobs for a playback service's typical output characteristics."""

    key: ServiceName
    display_name: str
    target_lufs: float
    bass_gain_db: float
    presence_gain_db: float
    air_gain_db: float
    stereo_width: float
    transient_restore: float


SERVICE_PROFILES: dict[ServiceName, ServiceProfile] = {
    ServiceName.SPOTIFY: ServiceProfile(
        key=ServiceName.SPOTIFY,
        display_name="Spotify",
        target_lufs=-15.0,
        bass_gain_db=0.8,
        presence_gain_db=0.9,
        air_gain_db=0.5,
        stereo_width=1.05,
        transient_restore=0.10,
    ),
    ServiceName.APPLE_MUSIC: ServiceProfile(
        key=ServiceName.APPLE_MUSIC,
        display_name="Apple Music",
        target_lufs=-16.0,
        bass_gain_db=0.4,
        presence_gain_db=0.5,
        air_gain_db=0.8,
        stereo_width=1.03,
        transient_restore=0.06,
    ),
    ServiceName.YOUTUBE_MUSIC: ServiceProfile(
        key=ServiceName.YOUTUBE_MUSIC,
        display_name="YouTube Music",
        target_lufs=-14.5,
        bass_gain_db=0.5,
        presence_gain_db=1.0,
        air_gain_db=0.4,
        stereo_width=1.02,
        transient_restore=0.12,
    ),
    ServiceName.GENERIC: ServiceProfile(
        key=ServiceName.GENERIC,
        display_name="Generic",
        target_lufs=-15.5,
        bass_gain_db=0.5,
        presence_gain_db=0.7,
        air_gain_db=0.5,
        stereo_width=1.03,
        transient_restore=0.08,
    ),
}


@dataclass(frozen=True)
class ListenerProfile:
    """Local-only personalization values derived from user preference."""

    bass_preference_db: float = 0.0
    vocal_clarity_db: float = 0.0
    brightness_db: float = 0.0
    loudness_bias_db: float = 0.0


def get_service_profile(service: str | ServiceName | None) -> ServiceProfile:
    """Return a service profile, falling back to generic for unknown apps."""

    if isinstance(service, ServiceName):
        return SERVICE_PROFILES[service]
    if not service:
        return SERVICE_PROFILES[ServiceName.GENERIC]

    normalized = service.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "apple": ServiceName.APPLE_MUSIC,
        "applemusic": ServiceName.APPLE_MUSIC,
        "itunes": ServiceName.APPLE_MUSIC,
        "youtube": ServiceName.YOUTUBE_MUSIC,
        "ytmusic": ServiceName.YOUTUBE_MUSIC,
        "you_tube_music": ServiceName.YOUTUBE_MUSIC,
        "spotify": ServiceName.SPOTIFY,
    }
    try:
        key = aliases.get(normalized, ServiceName(normalized))
    except ValueError:
        key = ServiceName.GENERIC
    return SERVICE_PROFILES[key]
