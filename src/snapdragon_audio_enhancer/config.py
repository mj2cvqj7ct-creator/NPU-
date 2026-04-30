from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any


class MusicService(str, Enum):
    """Services are hints for tuning only; their apps and streams are not modified."""

    GENERIC = "generic"
    SPOTIFY = "spotify"
    APPLE_MUSIC = "apple_music"
    YOUTUBE_MUSIC = "youtube_music"


@dataclass(frozen=True)
class ServiceProfile:
    service: MusicService
    target_lufs: float
    bass_gain_db: float
    presence_gain_db: float
    air_gain_db: float
    width: float
    npu_blend: float


SERVICE_PROFILES: dict[MusicService, ServiceProfile] = {
    MusicService.GENERIC: ServiceProfile(
        MusicService.GENERIC,
        target_lufs=-16.0,
        bass_gain_db=1.0,
        presence_gain_db=0.8,
        air_gain_db=0.8,
        width=1.03,
        npu_blend=0.45,
    ),
    MusicService.SPOTIFY: ServiceProfile(
        MusicService.SPOTIFY,
        target_lufs=-15.0,
        bass_gain_db=0.8,
        presence_gain_db=1.0,
        air_gain_db=0.9,
        width=1.04,
        npu_blend=0.5,
    ),
    MusicService.APPLE_MUSIC: ServiceProfile(
        MusicService.APPLE_MUSIC,
        target_lufs=-17.0,
        bass_gain_db=0.6,
        presence_gain_db=0.7,
        air_gain_db=0.7,
        width=1.02,
        npu_blend=0.35,
    ),
    MusicService.YOUTUBE_MUSIC: ServiceProfile(
        MusicService.YOUTUBE_MUSIC,
        target_lufs=-14.5,
        bass_gain_db=1.1,
        presence_gain_db=1.1,
        air_gain_db=1.0,
        width=1.03,
        npu_blend=0.55,
    ),
}


@dataclass(frozen=True)
class EnhancementConfig:
    sample_rate: int = 48_000
    frame_ms: float = 20.0
    service: MusicService = MusicService.GENERIC
    target_lufs: float = -16.0
    max_gain_db: float = 6.0
    true_peak_dbfs: float = -1.0
    bass_gain_db: float = 1.0
    presence_gain_db: float = 0.8
    air_gain_db: float = 0.8
    width: float = 1.03
    npu_blend: float = 0.45

    @property
    def frame_size(self) -> int:
        return max(1, round(self.sample_rate * self.frame_ms / 1000.0))

    @classmethod
    def for_service(
        cls,
        service: MusicService | str,
        *,
        sample_rate: int = 48_000,
        frame_ms: float = 20.0,
    ) -> "EnhancementConfig":
        parsed = MusicService(service)
        profile = SERVICE_PROFILES[parsed]
        return cls(
            sample_rate=sample_rate,
            frame_ms=frame_ms,
            service=parsed,
            target_lufs=profile.target_lufs,
            bass_gain_db=profile.bass_gain_db,
            presence_gain_db=profile.presence_gain_db,
            air_gain_db=profile.air_gain_db,
            width=profile.width,
            npu_blend=profile.npu_blend,
        )

    def with_profile(self, profile: ServiceProfile) -> "EnhancementConfig":
        return replace(
            self,
            service=profile.service,
            target_lufs=profile.target_lufs,
            bass_gain_db=profile.bass_gain_db,
            presence_gain_db=profile.presence_gain_db,
            air_gain_db=profile.air_gain_db,
            width=profile.width,
            npu_blend=profile.npu_blend,
        )

    def with_overrides(self, overrides: dict[str, Any]) -> "EnhancementConfig":
        allowed = set(self.to_dict())
        unknown = sorted(set(overrides) - allowed)
        if unknown:
            raise ValueError(f"Unknown enhancement setting(s): {', '.join(unknown)}")

        values = dict(overrides)
        if "service" in values:
            values["service"] = MusicService(values["service"])
        return replace(self, **values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "frame_ms": self.frame_ms,
            "service": self.service.value,
            "target_lufs": self.target_lufs,
            "max_gain_db": self.max_gain_db,
            "true_peak_dbfs": self.true_peak_dbfs,
            "bass_gain_db": self.bass_gain_db,
            "presence_gain_db": self.presence_gain_db,
            "air_gain_db": self.air_gain_db,
            "width": self.width,
            "npu_blend": self.npu_blend,
        }
