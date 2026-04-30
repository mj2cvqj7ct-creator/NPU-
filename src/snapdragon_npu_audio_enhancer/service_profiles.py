"""Service-aware tuning for streamed PCM enhancement.

The profiles never inspect or alter service internals. They only bias the
local DSP controls after the audio has reached the OS mixer as PCM.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .dsp import EnhancementControls


class MusicService(str, Enum):
    """Music services with conservative output-stage tuning."""

    GENERIC = "generic"
    SPOTIFY = "spotify"
    APPLE_MUSIC = "apple-music"
    YOUTUBE_MUSIC = "youtube-music"


@dataclass(frozen=True)
class ServiceProfile:
    """DSP/NPU control bias for a music-service output stream."""

    service: MusicService
    target_rms_db: float
    npu_mix: float
    bass_bias_db: float = 0.0
    presence_bias_db: float = 0.0
    air_bias_db: float = 0.0
    stereo_width_bias: float = 0.0
    compressor_threshold_bias_db: float = 0.0
    limiter_ceiling_db: float = -1.0

    def apply(self, controls: EnhancementControls) -> EnhancementControls:
        """Apply profile bias while preserving safe control bounds."""

        return EnhancementControls(
            pre_gain_db=controls.pre_gain_db,
            bass_gain_db=_clamp(controls.bass_gain_db + self.bass_bias_db, -3.0, 3.0),
            presence_gain_db=_clamp(controls.presence_gain_db + self.presence_bias_db, -3.0, 3.0),
            air_gain_db=_clamp(controls.air_gain_db + self.air_bias_db, -3.0, 3.0),
            stereo_width=_clamp(controls.stereo_width + self.stereo_width_bias, 0.5, 1.4),
            compressor_threshold_db=_clamp(
                controls.compressor_threshold_db + self.compressor_threshold_bias_db,
                -36.0,
                -6.0,
            ),
            compressor_ratio=controls.compressor_ratio,
            limiter_ceiling_db=min(controls.limiter_ceiling_db, self.limiter_ceiling_db),
        )


SERVICE_PROFILES: dict[MusicService, ServiceProfile] = {
    MusicService.GENERIC: ServiceProfile(
        service=MusicService.GENERIC,
        target_rms_db=-18.0,
        npu_mix=0.35,
    ),
    MusicService.SPOTIFY: ServiceProfile(
        service=MusicService.SPOTIFY,
        target_rms_db=-18.5,
        npu_mix=0.30,
        bass_bias_db=-0.2,
        presence_bias_db=0.2,
        air_bias_db=0.1,
        compressor_threshold_bias_db=1.0,
        limiter_ceiling_db=-1.1,
    ),
    MusicService.APPLE_MUSIC: ServiceProfile(
        service=MusicService.APPLE_MUSIC,
        target_rms_db=-19.0,
        npu_mix=0.25,
        bass_bias_db=0.1,
        presence_bias_db=0.1,
        air_bias_db=0.2,
        stereo_width_bias=0.01,
        compressor_threshold_bias_db=2.0,
        limiter_ceiling_db=-1.0,
    ),
    MusicService.YOUTUBE_MUSIC: ServiceProfile(
        service=MusicService.YOUTUBE_MUSIC,
        target_rms_db=-17.5,
        npu_mix=0.40,
        bass_bias_db=0.1,
        presence_bias_db=-0.2,
        air_bias_db=-0.1,
        stereo_width_bias=-0.01,
        compressor_threshold_bias_db=-1.0,
        limiter_ceiling_db=-1.2,
    ),
}


def get_service_profile(service: str | MusicService | None) -> ServiceProfile:
    """Return a profile for CLI/API input, defaulting to generic tuning."""

    if service is None:
        return SERVICE_PROFILES[MusicService.GENERIC]
    if isinstance(service, MusicService):
        return SERVICE_PROFILES[service]
    normalized = service.strip().lower().replace("_", "-")
    aliases = {
        "apple": MusicService.APPLE_MUSIC,
        "applemusic": MusicService.APPLE_MUSIC,
        "apple-music": MusicService.APPLE_MUSIC,
        "itunes": MusicService.APPLE_MUSIC,
        "spotify": MusicService.SPOTIFY,
        "youtube": MusicService.YOUTUBE_MUSIC,
        "youtubemusic": MusicService.YOUTUBE_MUSIC,
        "youtube-music": MusicService.YOUTUBE_MUSIC,
        "ytmusic": MusicService.YOUTUBE_MUSIC,
        "auto": MusicService.GENERIC,
        "generic": MusicService.GENERIC,
    }
    try:
        return SERVICE_PROFILES[aliases.get(normalized, MusicService(normalized))]
    except ValueError as exc:
        choices = ", ".join(service.value for service in MusicService)
        raise ValueError(f"Unsupported service '{service}'. Choose one of: {choices}.") from exc


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))
