"""Service-aware tuning for music app PCM output.

The enhancer never modifies Spotify, Apple Music, or YouTube Music internals.
These profiles only bias the local post-processing chain after WASAPI-style PCM
capture, matching the typical loudness and codec behavior of each source.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .dsp import AudioFeatures, EnhancementControls


class StreamingService(str, Enum):
    """Supported music-service tuning targets."""

    AUTO = "auto"
    SPOTIFY = "spotify"
    APPLE_MUSIC = "apple-music"
    YOUTUBE_MUSIC = "youtube-music"


@dataclass(frozen=True)
class ServiceProfile:
    """Controls how generic DSP/NPU output is biased for a service."""

    service: StreamingService
    display_name: str
    target_rms_db: float
    npu_mix: float
    bass_bias_db: float = 0.0
    presence_bias_db: float = 0.0
    air_bias_db: float = 0.0
    stereo_width_bias: float = 0.0
    compressor_threshold_db: float = -18.0
    compressor_ratio: float = 1.7
    limiter_ceiling_db: float = -1.0

    def apply(self, controls: EnhancementControls, features: AudioFeatures) -> EnhancementControls:
        """Return service-tuned controls clamped to safe real-time ranges."""

        pre_gain_db = controls.pre_gain_db
        if features.clipping_ratio > 0.001 or features.peak_db > -0.2:
            pre_gain_db = min(pre_gain_db, -1.5)

        return EnhancementControls(
            pre_gain_db=float(np.clip(pre_gain_db, -6.0, 6.0)),
            bass_gain_db=float(np.clip(controls.bass_gain_db + self.bass_bias_db, -3.0, 3.0)),
            presence_gain_db=float(np.clip(controls.presence_gain_db + self.presence_bias_db, -3.0, 3.0)),
            air_gain_db=float(np.clip(controls.air_gain_db + self.air_bias_db, -3.0, 3.0)),
            stereo_width=float(np.clip(controls.stereo_width + self.stereo_width_bias, 0.5, 1.35)),
            compressor_threshold_db=float(
                np.clip(
                    min(controls.compressor_threshold_db, self.compressor_threshold_db),
                    -36.0,
                    -6.0,
                )
            ),
            compressor_ratio=float(np.clip(max(controls.compressor_ratio, self.compressor_ratio), 1.0, 6.0)),
            limiter_ceiling_db=float(np.clip(min(controls.limiter_ceiling_db, self.limiter_ceiling_db), -6.0, -0.1)),
        )


SERVICE_PROFILES: dict[StreamingService, ServiceProfile] = {
    StreamingService.AUTO: ServiceProfile(
        service=StreamingService.AUTO,
        display_name="Auto / balanced",
        target_rms_db=-18.0,
        npu_mix=0.35,
        presence_bias_db=0.1,
        air_bias_db=0.1,
    ),
    StreamingService.SPOTIFY: ServiceProfile(
        service=StreamingService.SPOTIFY,
        display_name="Spotify",
        target_rms_db=-17.5,
        npu_mix=0.4,
        bass_bias_db=0.15,
        presence_bias_db=0.35,
        air_bias_db=0.45,
        stereo_width_bias=0.02,
        compressor_threshold_db=-17.0,
        compressor_ratio=1.6,
        limiter_ceiling_db=-1.2,
    ),
    StreamingService.APPLE_MUSIC: ServiceProfile(
        service=StreamingService.APPLE_MUSIC,
        display_name="Apple Music",
        target_rms_db=-18.5,
        npu_mix=0.28,
        bass_bias_db=0.1,
        presence_bias_db=0.15,
        air_bias_db=0.25,
        stereo_width_bias=0.01,
        compressor_threshold_db=-14.0,
        compressor_ratio=1.35,
        limiter_ceiling_db=-1.0,
    ),
    StreamingService.YOUTUBE_MUSIC: ServiceProfile(
        service=StreamingService.YOUTUBE_MUSIC,
        display_name="YouTube Music",
        target_rms_db=-18.0,
        npu_mix=0.45,
        bass_bias_db=-0.05,
        presence_bias_db=0.45,
        air_bias_db=0.35,
        stereo_width_bias=0.03,
        compressor_threshold_db=-20.0,
        compressor_ratio=2.0,
        limiter_ceiling_db=-1.5,
    ),
}


def get_service_profile(service: StreamingService | str | None = None) -> ServiceProfile:
    """Resolve a service name to a safe enhancement profile."""

    if service is None:
        return SERVICE_PROFILES[StreamingService.AUTO]
    if isinstance(service, StreamingService):
        return SERVICE_PROFILES[service]

    aliases = {
        "apple": StreamingService.APPLE_MUSIC,
        "apple_music": StreamingService.APPLE_MUSIC,
        "applemusic": StreamingService.APPLE_MUSIC,
        "youtube": StreamingService.YOUTUBE_MUSIC,
        "youtube_music": StreamingService.YOUTUBE_MUSIC,
        "youtubemusic": StreamingService.YOUTUBE_MUSIC,
        "ytmusic": StreamingService.YOUTUBE_MUSIC,
    }
    normalized = str(service).strip().lower()
    try:
        key = aliases[normalized] if normalized in aliases else StreamingService(normalized)
    except ValueError as exc:
        supported = ", ".join(item.value for item in StreamingService)
        raise ValueError(f"Unsupported service '{service}'. Choose one of: {supported}") from exc
    return SERVICE_PROFILES[key]
