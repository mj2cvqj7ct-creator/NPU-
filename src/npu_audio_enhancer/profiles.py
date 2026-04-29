from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class StreamingService(StrEnum):
    SPOTIFY = "spotify"
    APPLE_MUSIC = "apple_music"
    YOUTUBE_MUSIC = "youtube_music"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class EnhancementProfile:
    service: StreamingService
    headphone_profile: str
    target_rms: float
    max_gain_db: float
    low_shelf: float
    presence_boost: float
    air_boost: float
    stereo_width: float


PROFILES: dict[StreamingService, EnhancementProfile] = {
    StreamingService.SPOTIFY: EnhancementProfile(
        service=StreamingService.SPOTIFY,
        headphone_profile="generic",
        target_rms=0.145,
        max_gain_db=5.0,
        low_shelf=0.035,
        presence_boost=0.03,
        air_boost=0.015,
        stereo_width=1.02,
    ),
    StreamingService.APPLE_MUSIC: EnhancementProfile(
        service=StreamingService.APPLE_MUSIC,
        headphone_profile="generic",
        target_rms=0.135,
        max_gain_db=4.0,
        low_shelf=0.015,
        presence_boost=0.02,
        air_boost=0.01,
        stereo_width=1.01,
    ),
    StreamingService.YOUTUBE_MUSIC: EnhancementProfile(
        service=StreamingService.YOUTUBE_MUSIC,
        headphone_profile="generic",
        target_rms=0.15,
        max_gain_db=5.5,
        low_shelf=0.025,
        presence_boost=0.045,
        air_boost=0.02,
        stereo_width=1.0,
    ),
    StreamingService.NEUTRAL: EnhancementProfile(
        service=StreamingService.NEUTRAL,
        headphone_profile="generic",
        target_rms=0.14,
        max_gain_db=4.5,
        low_shelf=0.02,
        presence_boost=0.025,
        air_boost=0.012,
        stereo_width=1.0,
    ),
}


def service_profile(service: str, headphone_profile: str = "generic") -> EnhancementProfile:
    normalized = service.strip().lower().replace("-", "_")
    try:
        service_id = StreamingService(normalized)
    except ValueError:
        service_id = StreamingService.NEUTRAL

    base = PROFILES[service_id]
    headphone = headphone_profile.strip().lower() or "generic"
    if headphone == "bright":
        return EnhancementProfile(
            service=base.service,
            headphone_profile=headphone,
            target_rms=base.target_rms,
            max_gain_db=base.max_gain_db,
            low_shelf=base.low_shelf + 0.01,
            presence_boost=max(0.0, base.presence_boost - 0.015),
            air_boost=max(0.0, base.air_boost - 0.01),
            stereo_width=base.stereo_width,
        )

    return EnhancementProfile(
        service=base.service,
        headphone_profile=headphone,
        target_rms=base.target_rms,
        max_gain_db=base.max_gain_db,
        low_shelf=base.low_shelf,
        presence_boost=base.presence_boost,
        air_boost=base.air_boost,
        stereo_width=base.stereo_width,
    )
