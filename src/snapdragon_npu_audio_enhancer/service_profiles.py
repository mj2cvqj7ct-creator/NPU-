"""Service-aware tuning for local streaming audio enhancement.

The official Spotify, Apple Music, and YouTube Music clients are not modified.
These profiles only steer the local PCM post-processing chain after audio has
already reached the operating system mixer or a loopback capture path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .dsp import AudioFeatures, EnhancementControls


class StreamingService(str, Enum):
    """Supported service hints for local post-processing."""

    NEUTRAL = "neutral"
    SPOTIFY = "spotify"
    APPLE_MUSIC = "apple-music"
    YOUTUBE_MUSIC = "youtube-music"


@dataclass(frozen=True)
class ServiceEnhancementProfile:
    """Controls how the shared DSP/NPU chain is biased for a service."""

    service: StreamingService
    target_rms_db: float
    npu_mix: float
    bass_bias_db: float = 0.0
    presence_bias_db: float = 0.0
    air_bias_db: float = 0.0
    stereo_width_bias: float = 0.0
    compressor_ratio_scale: float = 1.0
    limiter_ceiling_db: float = -1.0
    clipping_pre_gain_db: float = -1.5

    def tune_controls(self, controls: EnhancementControls, features: AudioFeatures) -> EnhancementControls:
        """Return service-biased controls while preserving limiter safety."""

        bass_bias = self.bass_bias_db
        presence_bias = self.presence_bias_db
        air_bias = self.air_bias_db
        stereo_width_bias = self.stereo_width_bias
        pre_gain_db = controls.pre_gain_db

        if self.service == StreamingService.SPOTIFY:
            # Often already loud and compressed: add clarity only when the band is not crowded.
            presence_bias += float(np.clip(0.7 - features.vocal_band_energy * 4.0, -0.2, 0.5))
            air_bias += float(np.clip(0.6 - features.high_band_energy * 10.0, -0.1, 0.4))
            bass_bias += float(np.clip(0.5 - features.low_band_energy * 6.0, -0.2, 0.4))
        elif self.service == StreamingService.APPLE_MUSIC:
            # Lossless streams benefit from restraint: widen and brighten less, compress less.
            stereo_width_bias += 0.02 if features.stereo_correlation > 0.2 else 0.0
            air_bias += float(np.clip(0.4 - features.high_band_energy * 8.0, -0.1, 0.25))
        elif self.service == StreamingService.YOUTUBE_MUSIC:
            # Browser/app output can vary widely in loudness and harshness.
            if features.peak_db > -1.0 or features.clipping_ratio > 0.0005:
                pre_gain_db = min(pre_gain_db, self.clipping_pre_gain_db)
            presence_bias += float(np.clip(0.5 - features.vocal_band_energy * 6.0, -0.4, 0.35))
            stereo_width_bias -= 0.02 if features.stereo_correlation < 0.1 else 0.0

        return EnhancementControls(
            pre_gain_db=float(np.clip(pre_gain_db, -6.0, 6.0)),
            bass_gain_db=float(np.clip(controls.bass_gain_db + bass_bias, -3.0, 3.0)),
            presence_gain_db=float(np.clip(controls.presence_gain_db + presence_bias, -3.0, 3.0)),
            air_gain_db=float(np.clip(controls.air_gain_db + air_bias, -3.0, 3.0)),
            stereo_width=float(np.clip(controls.stereo_width + stereo_width_bias, 0.5, 1.4)),
            compressor_threshold_db=controls.compressor_threshold_db,
            compressor_ratio=float(np.clip(controls.compressor_ratio * self.compressor_ratio_scale, 1.0, 6.0)),
            limiter_ceiling_db=min(controls.limiter_ceiling_db, self.limiter_ceiling_db),
        )


SERVICE_PROFILES: dict[StreamingService, ServiceEnhancementProfile] = {
    StreamingService.NEUTRAL: ServiceEnhancementProfile(
        service=StreamingService.NEUTRAL,
        target_rms_db=-16.0,
        npu_mix=0.35,
    ),
    StreamingService.SPOTIFY: ServiceEnhancementProfile(
        service=StreamingService.SPOTIFY,
        target_rms_db=-17.0,
        npu_mix=0.45,
        bass_bias_db=0.15,
        presence_bias_db=0.2,
        air_bias_db=0.1,
        stereo_width_bias=0.02,
        compressor_ratio_scale=0.95,
        limiter_ceiling_db=-1.0,
    ),
    StreamingService.APPLE_MUSIC: ServiceEnhancementProfile(
        service=StreamingService.APPLE_MUSIC,
        target_rms_db=-18.0,
        npu_mix=0.28,
        bass_bias_db=0.05,
        presence_bias_db=0.05,
        air_bias_db=0.1,
        stereo_width_bias=0.01,
        compressor_ratio_scale=0.85,
        limiter_ceiling_db=-1.0,
    ),
    StreamingService.YOUTUBE_MUSIC: ServiceEnhancementProfile(
        service=StreamingService.YOUTUBE_MUSIC,
        target_rms_db=-17.5,
        npu_mix=0.4,
        bass_bias_db=0.1,
        presence_bias_db=0.05,
        air_bias_db=-0.05,
        compressor_ratio_scale=1.05,
        limiter_ceiling_db=-1.5,
        clipping_pre_gain_db=-2.0,
    ),
}


def get_service_profile(service: str | StreamingService | None) -> ServiceEnhancementProfile:
    """Resolve a service hint to a profile."""

    if service is None:
        return SERVICE_PROFILES[StreamingService.NEUTRAL]
    if isinstance(service, StreamingService):
        return SERVICE_PROFILES[service]

    normalized = service.strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "apple": StreamingService.APPLE_MUSIC,
        "applemusic": StreamingService.APPLE_MUSIC,
        "apple-music": StreamingService.APPLE_MUSIC,
        "spotify": StreamingService.SPOTIFY,
        "youtube": StreamingService.YOUTUBE_MUSIC,
        "youtubemusic": StreamingService.YOUTUBE_MUSIC,
        "youtube-music": StreamingService.YOUTUBE_MUSIC,
        "ytmusic": StreamingService.YOUTUBE_MUSIC,
        "neutral": StreamingService.NEUTRAL,
        "generic": StreamingService.NEUTRAL,
    }
    try:
        return SERVICE_PROFILES[aliases[normalized]]
    except KeyError as exc:
        valid = ", ".join(service.value for service in StreamingService)
        raise ValueError(f"unknown service '{service}'. Expected one of: {valid}") from exc
