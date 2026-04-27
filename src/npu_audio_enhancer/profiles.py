from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnhancementProfile:
    """Service and device tuning values for adaptive enhancement."""

    name: str = "balanced"
    target_rms_db: float = -18.0
    max_gain_db: float = 9.0
    bass_boost_db: float = 1.5
    clarity_boost_db: float = 1.25
    compression_strength: float = 0.25
    limiter_ceiling_db: float = -1.0
    stereo_width_limit: float = 1.12


def service_profile(service: str | None) -> EnhancementProfile:
    key = (service or "balanced").strip().lower().replace("_", "-")
    profiles = {
        "spotify": EnhancementProfile(
            name="spotify",
            target_rms_db=-17.0,
            max_gain_db=7.5,
            bass_boost_db=1.25,
            clarity_boost_db=1.5,
            compression_strength=0.18,
        ),
        "apple": EnhancementProfile(
            name="apple-music",
            target_rms_db=-19.0,
            max_gain_db=6.0,
            bass_boost_db=1.0,
            clarity_boost_db=1.0,
            compression_strength=0.12,
            limiter_ceiling_db=-1.2,
        ),
        "apple-music": EnhancementProfile(
            name="apple-music",
            target_rms_db=-19.0,
            max_gain_db=6.0,
            bass_boost_db=1.0,
            clarity_boost_db=1.0,
            compression_strength=0.12,
            limiter_ceiling_db=-1.2,
        ),
        "youtube": EnhancementProfile(
            name="youtube-music",
            target_rms_db=-18.0,
            max_gain_db=8.0,
            bass_boost_db=1.5,
            clarity_boost_db=1.75,
            compression_strength=0.22,
        ),
        "youtube-music": EnhancementProfile(
            name="youtube-music",
            target_rms_db=-18.0,
            max_gain_db=8.0,
            bass_boost_db=1.5,
            clarity_boost_db=1.75,
            compression_strength=0.22,
        ),
    }
    return profiles.get(key, EnhancementProfile())
