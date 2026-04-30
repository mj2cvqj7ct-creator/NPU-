from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MusicService(str, Enum):
    SPOTIFY = "spotify"
    APPLE_MUSIC = "apple_music"
    YOUTUBE_MUSIC = "youtube_music"
    GENERIC = "generic"

    @classmethod
    def parse(cls, value: str | None) -> "MusicService":
        if value is None:
            return cls.GENERIC

        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "apple": cls.APPLE_MUSIC,
            "applemusic": cls.APPLE_MUSIC,
            "apple_music": cls.APPLE_MUSIC,
            "itunes": cls.APPLE_MUSIC,
            "spotify": cls.SPOTIFY,
            "youtube": cls.YOUTUBE_MUSIC,
            "youtube_music": cls.YOUTUBE_MUSIC,
            "youtubemusic": cls.YOUTUBE_MUSIC,
            "ytmusic": cls.YOUTUBE_MUSIC,
        }
        return aliases.get(normalized, cls.GENERIC)


@dataclass(frozen=True)
class ServicePolicy:
    service: MusicService
    loudness_target_lufs: float
    clarity: float
    bass_weight: float
    stereo_width: float
    limiter_ceiling_dbfs: float = -1.0


POLICIES: dict[MusicService, ServicePolicy] = {
    MusicService.SPOTIFY: ServicePolicy(
        service=MusicService.SPOTIFY,
        loudness_target_lufs=-15.0,
        clarity=0.52,
        bass_weight=0.18,
        stereo_width=0.05,
    ),
    MusicService.APPLE_MUSIC: ServicePolicy(
        service=MusicService.APPLE_MUSIC,
        loudness_target_lufs=-16.0,
        clarity=0.34,
        bass_weight=0.12,
        stereo_width=0.03,
    ),
    MusicService.YOUTUBE_MUSIC: ServicePolicy(
        service=MusicService.YOUTUBE_MUSIC,
        loudness_target_lufs=-14.0,
        clarity=0.46,
        bass_weight=0.14,
        stereo_width=0.02,
    ),
    MusicService.GENERIC: ServicePolicy(
        service=MusicService.GENERIC,
        loudness_target_lufs=-15.0,
        clarity=0.40,
        bass_weight=0.12,
        stereo_width=0.02,
    ),
}


def get_policy(service: MusicService | str | None) -> ServicePolicy:
    parsed = service if isinstance(service, MusicService) else MusicService.parse(service)
    return POLICIES[parsed]
