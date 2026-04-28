"""Service-agnostic enhancement profiles.

The profiles tune the local post-processing chain for the typical delivery
characteristics of each player. They do not rely on private APIs or modify app
behavior.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ServiceProfile:
    """Local audio-shaping defaults for a playback source."""

    name: str
    target_lufs: float
    warmth_db: float
    clarity_db: float
    transient_restore: float


_PROFILES = {
    "spotify": ServiceProfile(
        name="spotify",
        target_lufs=-15.0,
        warmth_db=0.8,
        clarity_db=1.2,
        transient_restore=0.25,
    ),
    "apple_music": ServiceProfile(
        name="apple_music",
        target_lufs=-16.0,
        warmth_db=0.5,
        clarity_db=0.9,
        transient_restore=0.15,
    ),
    "youtube_music": ServiceProfile(
        name="youtube_music",
        target_lufs=-14.0,
        warmth_db=0.6,
        clarity_db=1.4,
        transient_restore=0.3,
    ),
    "generic": ServiceProfile(
        name="generic",
        target_lufs=-15.5,
        warmth_db=0.5,
        clarity_db=1.0,
        transient_restore=0.2,
    ),
}


def get_service_profile(service: str | None) -> ServiceProfile:
    """Return a conservative local profile for a music service name."""

    if service is None:
        return _PROFILES["generic"]

    key = service.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "apple": "apple_music",
        "itunes": "apple_music",
        "youtube": "youtube_music",
        "ytmusic": "youtube_music",
        "yt_music": "youtube_music",
    }
    return _PROFILES.get(aliases.get(key, key), _PROFILES["generic"])
