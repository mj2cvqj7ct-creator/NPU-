from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MusicService(str, Enum):
    """Supported source presets for service-specific local post-processing."""

    AUTO = "auto"
    SPOTIFY = "spotify"
    APPLE_MUSIC = "apple_music"
    YOUTUBE_MUSIC = "youtube_music"


@dataclass(frozen=True)
class ServiceProfile:
    """Tuning weights for locally rendered PCM from a streaming service."""

    service: MusicService
    target_loudness_db: float
    loudness_bias_db: float = 0.0
    bass_weight: float = 1.0
    vocal_clarity_weight: float = 1.0
    air_weight: float = 1.0
    width_weight: float = 1.0
    transient_weight: float = 1.0
    bass_bias_db: float = 0.0
    presence_bias_db: float = 0.0
    air_bias_db: float = 0.0
    stereo_width: float = 1.06
    low_band_reference: float = 0.16
    vocal_reference: float = 0.22
    compressor_threshold_db: float = -16.0
    compressor_ratio: float = 1.55
    compressor_bias_db: float = 0.0
    limiter_ceiling_db: float = -1.0
    npu_mix: float = 0.35

    @classmethod
    def from_name(cls, value: str | MusicService) -> "ServiceProfile":
        service = value if isinstance(value, MusicService) else MusicService(value)
        return SERVICE_PROFILES[service]


SERVICE_PROFILES: dict[MusicService, ServiceProfile] = {
    MusicService.AUTO: ServiceProfile(
        service=MusicService.AUTO,
        target_loudness_db=-18.0,
        npu_mix=0.35,
    ),
    MusicService.SPOTIFY: ServiceProfile(
        service=MusicService.SPOTIFY,
        target_loudness_db=-17.5,
        bass_weight=1.08,
        vocal_clarity_weight=1.05,
        air_weight=0.96,
        width_weight=0.9,
        transient_weight=1.15,
        bass_bias_db=0.4,
        presence_bias_db=0.3,
        air_bias_db=0.2,
        low_band_reference=0.17,
        vocal_reference=0.24,
        npu_mix=0.4,
    ),
    MusicService.APPLE_MUSIC: ServiceProfile(
        service=MusicService.APPLE_MUSIC,
        target_loudness_db=-18.5,
        bass_weight=0.95,
        vocal_clarity_weight=0.95,
        air_weight=1.12,
        width_weight=1.05,
        transient_weight=0.9,
        bass_bias_db=0.2,
        presence_bias_db=0.1,
        air_bias_db=0.35,
        stereo_width=1.08,
        low_band_reference=0.15,
        vocal_reference=0.21,
        npu_mix=0.3,
    ),
    MusicService.YOUTUBE_MUSIC: ServiceProfile(
        service=MusicService.YOUTUBE_MUSIC,
        target_loudness_db=-18.0,
        bass_weight=1.0,
        vocal_clarity_weight=1.18,
        air_weight=0.9,
        width_weight=0.75,
        transient_weight=1.25,
        bass_bias_db=0.1,
        presence_bias_db=0.45,
        air_bias_db=0.25,
        stereo_width=1.02,
        low_band_reference=0.15,
        vocal_reference=0.25,
        compressor_threshold_db=-17.0,
        npu_mix=0.45,
    ),
}


def get_service_profile(service: str | MusicService, _features: object | None = None) -> ServiceProfile:
    """Resolve a service preset.

    AUTO currently uses neutral tuning. The feature argument keeps the API ready
    for future local source detection without touching streaming-service internals.
    """

    if isinstance(service, str):
        service = MusicService(service)
    return SERVICE_PROFILES[service]
