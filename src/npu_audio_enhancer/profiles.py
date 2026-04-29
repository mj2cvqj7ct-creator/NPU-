from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ServiceProfileName(StrEnum):
    """Supported service profile identifiers for post-render PCM audio."""

    SPOTIFY = "spotify"
    APPLE_MUSIC = "apple_music"
    YOUTUBE_MUSIC = "youtube_music"
    GENERIC = "generic"


@dataclass(frozen=True)
class DynamicEqProfile:
    """Small EQ moves that stay safe for always-on real-time playback."""

    bass_gain_db: float
    presence_gain_db: float
    air_gain_db: float
    stereo_width: float
    transient_restore: float


@dataclass(frozen=True)
class ServiceProfile:
    """Conservative tuning values applied after app audio reaches the OS mixer."""

    service_name: str
    target_lufs: float
    max_gain_db: float
    limiter_ceiling_db: float
    eq: DynamicEqProfile

    def validate(self) -> None:
        if not -24.0 <= self.target_lufs <= -10.0:
            raise ValueError("target_lufs must stay in a safe music playback range")
        if not 0.0 <= self.max_gain_db <= 9.0:
            raise ValueError("max_gain_db must prevent aggressive loudness jumps")
        if not -3.0 <= self.limiter_ceiling_db <= -0.1:
            raise ValueError("limiter_ceiling_db must leave true-peak headroom")
        if not 0.7 <= self.eq.stereo_width <= 1.25:
            raise ValueError("stereo_width must avoid destructive widening")
        if not 0.0 <= self.eq.transient_restore <= 1.0:
            raise ValueError("transient_restore must be between 0 and 1")


SERVICE_PROFILES: dict[ServiceProfileName, ServiceProfile] = {
    ServiceProfileName.SPOTIFY: ServiceProfile(
        service_name=ServiceProfileName.SPOTIFY.value,
        target_lufs=-15.0,
        max_gain_db=4.5,
        limiter_ceiling_db=-1.0,
        eq=DynamicEqProfile(
            bass_gain_db=0.8,
            presence_gain_db=1.0,
            air_gain_db=0.6,
            stereo_width=1.04,
            transient_restore=0.25,
        ),
    ),
    ServiceProfileName.APPLE_MUSIC: ServiceProfile(
        service_name=ServiceProfileName.APPLE_MUSIC.value,
        target_lufs=-16.0,
        max_gain_db=3.0,
        limiter_ceiling_db=-1.0,
        eq=DynamicEqProfile(
            bass_gain_db=0.4,
            presence_gain_db=0.5,
            air_gain_db=0.8,
            stereo_width=1.02,
            transient_restore=0.15,
        ),
    ),
    ServiceProfileName.YOUTUBE_MUSIC: ServiceProfile(
        service_name=ServiceProfileName.YOUTUBE_MUSIC.value,
        target_lufs=-14.5,
        max_gain_db=5.0,
        limiter_ceiling_db=-1.0,
        eq=DynamicEqProfile(
            bass_gain_db=0.7,
            presence_gain_db=1.2,
            air_gain_db=0.4,
            stereo_width=1.0,
            transient_restore=0.3,
        ),
    ),
    ServiceProfileName.GENERIC: ServiceProfile(
        service_name=ServiceProfileName.GENERIC.value,
        target_lufs=-15.5,
        max_gain_db=4.0,
        limiter_ceiling_db=-1.0,
        eq=DynamicEqProfile(
            bass_gain_db=0.5,
            presence_gain_db=0.7,
            air_gain_db=0.5,
            stereo_width=1.0,
            transient_restore=0.2,
        ),
    ),
}


def get_service_profile(service: ServiceProfileName | str) -> ServiceProfile:
    resolved = ServiceProfileName(service)
    profile = SERVICE_PROFILES[resolved]
    profile.validate()
    return profile
