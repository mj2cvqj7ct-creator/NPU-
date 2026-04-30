from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import pow


class MusicService(StrEnum):
    """Known playback sources supported by the local enhancer."""

    SPOTIFY = "spotify"
    APPLE_MUSIC = "apple_music"
    YOUTUBE_MUSIC = "youtube_music"
    GENERIC = "generic"


@dataclass(frozen=True)
class EnhancementProfile:
    """Tuning parameters for a local, service-agnostic enhancement pass."""

    service: MusicService
    target_lufs: float
    bass_gain_db: float
    presence_gain_db: float
    air_gain_db: float
    stereo_width: float
    limiter_ceiling: float = 0.98

    @property
    def target_rms(self) -> float:
        # Approximate LUFS target as RMS amplitude for this lightweight prototype.
        return pow(10.0, self.target_lufs / 20.0)

    @property
    def bass_gain(self) -> float:
        return _db_to_linear_delta(self.bass_gain_db)

    @property
    def presence_gain(self) -> float:
        return _db_to_linear_delta(self.presence_gain_db)

    @property
    def air_gain(self) -> float:
        return _db_to_linear_delta(self.air_gain_db)

    @classmethod
    def for_service(cls, service: MusicService) -> "EnhancementProfile":
        if service is MusicService.SPOTIFY:
            return cls(
                service=service,
                target_lufs=-15.0,
                bass_gain_db=1.4,
                presence_gain_db=1.1,
                air_gain_db=0.8,
                stereo_width=1.04,
            )
        if service is MusicService.APPLE_MUSIC:
            return cls(
                service=service,
                target_lufs=-16.0,
                bass_gain_db=0.8,
                presence_gain_db=0.7,
                air_gain_db=0.5,
                stereo_width=1.02,
            )
        if service is MusicService.YOUTUBE_MUSIC:
            return cls(
                service=service,
                target_lufs=-14.0,
                bass_gain_db=1.2,
                presence_gain_db=1.3,
                air_gain_db=0.9,
                stereo_width=1.03,
            )
        return cls(
            service=MusicService.GENERIC,
            target_lufs=-15.5,
            bass_gain_db=1.0,
            presence_gain_db=1.0,
            air_gain_db=0.6,
            stereo_width=1.02,
        )


def parse_service(value: str) -> MusicService:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "apple": MusicService.APPLE_MUSIC,
        "itunes": MusicService.APPLE_MUSIC,
        "youtube": MusicService.YOUTUBE_MUSIC,
        "ytmusic": MusicService.YOUTUBE_MUSIC,
        "yt_music": MusicService.YOUTUBE_MUSIC,
    }
    if normalized in aliases:
        return aliases[normalized]
    try:
        return MusicService(normalized)
    except ValueError as exc:
        supported = ", ".join(service.value for service in MusicService)
        raise ValueError(f"Unsupported service '{value}'. Use one of: {supported}.") from exc


def _db_to_linear_delta(db_value: float) -> float:
    return pow(10.0, db_value / 20.0) - 1.0
