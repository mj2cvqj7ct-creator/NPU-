from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ServiceProfile:
    """Safe default tuning for a family of streaming playback output."""

    name: str
    target_loudness_db: float
    bass_tilt_db: float
    presence_db: float
    air_db: float
    compression_ratio: float
    stereo_width: float
    transient_restore: float
    true_peak_ceiling: float


SERVICE_PROFILES: dict[str, ServiceProfile] = {
    "spotify": ServiceProfile(
        name="spotify",
        target_loudness_db=-16.0,
        bass_tilt_db=1.2,
        presence_db=1.6,
        air_db=0.8,
        compression_ratio=1.35,
        stereo_width=1.08,
        transient_restore=0.22,
        true_peak_ceiling=0.98,
    ),
    "apple_music": ServiceProfile(
        name="apple_music",
        target_loudness_db=-18.0,
        bass_tilt_db=0.8,
        presence_db=1.0,
        air_db=0.9,
        compression_ratio=1.18,
        stereo_width=1.04,
        transient_restore=0.14,
        true_peak_ceiling=0.98,
    ),
    "youtube_music": ServiceProfile(
        name="youtube_music",
        target_loudness_db=-15.0,
        bass_tilt_db=1.0,
        presence_db=1.8,
        air_db=0.6,
        compression_ratio=1.45,
        stereo_width=1.06,
        transient_restore=0.18,
        true_peak_ceiling=0.97,
    ),
    "generic": ServiceProfile(
        name="generic",
        target_loudness_db=-16.0,
        bass_tilt_db=0.8,
        presence_db=1.2,
        air_db=0.6,
        compression_ratio=1.3,
        stereo_width=1.04,
        transient_restore=0.15,
        true_peak_ceiling=0.98,
    ),
}


def resolve_service_profile(service: str) -> ServiceProfile:
    normalized = service.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "apple": "apple_music",
        "applemusic": "apple_music",
        "apple_music": "apple_music",
        "youtube": "youtube_music",
        "youtube_music": "youtube_music",
        "youtubemusic": "youtube_music",
        "spotify": "spotify",
    }
    return SERVICE_PROFILES[aliases.get(normalized, "generic")]


get_service_profile = resolve_service_profile
