from __future__ import annotations

from dataclasses import dataclass

from .models import StreamingService


@dataclass(frozen=True)
class ServiceProfile:
    """Safe per-service tone offsets for post-processing decoded PCM only."""

    clarity: float = 1.0
    warmth: float = 1.0
    stereo_width: float = 1.0
    loudness_offset_db: float = 0.0


_PROFILES: dict[StreamingService, ServiceProfile] = {
    StreamingService.GENERIC: ServiceProfile(),
    StreamingService.SPOTIFY: ServiceProfile(clarity=1.08, warmth=0.98, stereo_width=1.02),
    StreamingService.APPLE_MUSIC: ServiceProfile(clarity=1.03, warmth=1.03, stereo_width=1.0, loudness_offset_db=-1.0),
    StreamingService.YOUTUBE_MUSIC: ServiceProfile(clarity=1.1, warmth=0.96, stereo_width=0.98, loudness_offset_db=-0.5),
}


def resolve_service_profile(service: StreamingService) -> ServiceProfile:
    return _PROFILES.get(service, _PROFILES[StreamingService.GENERIC])
