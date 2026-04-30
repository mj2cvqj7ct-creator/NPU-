"""Service-facing profile names for supported music apps."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnhancementProfile:
    """Per-service tuning that stays within OS-level audio post-processing."""

    service: str
    target_rms_dbfs: float
    max_gain_db: float
    presence_gain_db: float
    bass_tighten: float
    stereo_width: float
    limiter_ceiling: float = 0.98


SPOTIFY_PROFILE = EnhancementProfile(
    service="spotify",
    target_rms_dbfs=-15.0,
    max_gain_db=4.0,
    presence_gain_db=1.2,
    bass_tighten=0.18,
    stereo_width=1.04,
)

APPLE_MUSIC_PROFILE = EnhancementProfile(
    service="apple_music",
    target_rms_dbfs=-17.0,
    max_gain_db=3.0,
    presence_gain_db=0.7,
    bass_tighten=0.08,
    stereo_width=1.02,
)

YOUTUBE_MUSIC_PROFILE = EnhancementProfile(
    service="youtube_music",
    target_rms_dbfs=-16.0,
    max_gain_db=5.0,
    presence_gain_db=1.5,
    bass_tighten=0.24,
    stereo_width=1.03,
)

GENERIC_PROFILE = EnhancementProfile(
    service="generic",
    target_rms_dbfs=-16.0,
    max_gain_db=3.5,
    presence_gain_db=1.0,
    bass_tighten=0.12,
    stereo_width=1.02,
)


@dataclass(frozen=True)
class ServiceProfile:
    """Public profile metadata used by UI, CLI, and capture integrations."""

    key: str
    display_name: str
    dsp_profile: EnhancementProfile
    capture_hint: str


SERVICE_PROFILES = {
    "spotify": ServiceProfile(
        key="spotify",
        display_name="Spotify",
        dsp_profile=SPOTIFY_PROFILE,
        capture_hint="Match Spotify.exe audio sessions when available; otherwise use loopback mix.",
    ),
    "apple_music": ServiceProfile(
        key="apple_music",
        display_name="Apple Music",
        dsp_profile=APPLE_MUSIC_PROFILE,
        capture_hint="Prefer AppleMusic.exe or iTunes.exe sessions; preserve lossless headroom.",
    ),
    "youtube_music": ServiceProfile(
        key="youtube_music",
        display_name="YouTube Music",
        dsp_profile=YOUTUBE_MUSIC_PROFILE,
        capture_hint="Match browser PWA/tab audio through WASAPI loopback session metadata.",
    ),
    "generic": ServiceProfile(
        key="generic",
        display_name="Generic music app",
        dsp_profile=GENERIC_PROFILE,
        capture_hint="Use the default render endpoint loopback mix.",
    ),
}

ALIASES = {
    "apple": "apple_music",
    "applemusic": "apple_music",
    "itunes": "apple_music",
    "youtube": "youtube_music",
    "ytmusic": "youtube_music",
    "youtube_music": "youtube_music",
    "youtube-music": "youtube_music",
}


def normalize_service_key(service: str | None) -> str:
    if not service:
        return "generic"
    key = service.strip().lower().replace(" ", "_")
    return ALIASES.get(key, key)


def get_service_profile(service: str | None) -> ServiceProfile:
    return SERVICE_PROFILES.get(normalize_service_key(service), SERVICE_PROFILES["generic"])
