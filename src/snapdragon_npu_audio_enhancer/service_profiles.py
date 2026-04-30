from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dsp import EnhancementControls


@dataclass(frozen=True)
class ServiceEnhancementProfile:
    """Service-aware PCM post-processing policy.

    The profile never assumes access to a streaming service's private data. It
    only biases the local DSP/NPU controls after the app has rendered PCM to the
    operating system audio stack.
    """

    name: str
    display_name: str
    description: str
    target_rms_db: float
    control_bias: EnhancementControls

    def apply(self, controls: EnhancementControls) -> EnhancementControls:
        bias = self.control_bias
        return EnhancementControls(
            pre_gain_db=float(np.clip(controls.pre_gain_db + bias.pre_gain_db, -6.0, 6.0)),
            bass_gain_db=float(np.clip(controls.bass_gain_db + bias.bass_gain_db, -3.0, 3.0)),
            presence_gain_db=float(
                np.clip(controls.presence_gain_db + bias.presence_gain_db, -3.0, 3.0)
            ),
            air_gain_db=float(np.clip(controls.air_gain_db + bias.air_gain_db, -3.0, 3.0)),
            stereo_width=float(np.clip(controls.stereo_width * bias.stereo_width, 0.5, 1.4)),
            compressor_threshold_db=float(
                np.clip(
                    controls.compressor_threshold_db + bias.compressor_threshold_db,
                    -36.0,
                    -6.0,
                )
            ),
            compressor_ratio=float(np.clip(controls.compressor_ratio + bias.compressor_ratio, 1.0, 6.0)),
            limiter_ceiling_db=float(
                np.clip(controls.limiter_ceiling_db + bias.limiter_ceiling_db, -6.0, -0.1)
            ),
        )


NEUTRAL_PROFILE = ServiceEnhancementProfile(
    name="neutral",
    display_name="Neutral streaming",
    description="Balanced low-latency enhancement for generic PCM output.",
    target_rms_db=-18.0,
    control_bias=EnhancementControls(
        stereo_width=1.0,
        compressor_threshold_db=0.0,
        compressor_ratio=0.0,
        limiter_ceiling_db=0.0,
    ),
)


SERVICE_PROFILES: dict[str, ServiceEnhancementProfile] = {
    "neutral": NEUTRAL_PROFILE,
    "spotify": ServiceEnhancementProfile(
        name="spotify",
        display_name="Spotify",
        description=(
            "Tightens bass, protects already-normalized masters, and adds "
            "moderate vocal presence for compressed streaming output."
        ),
        target_rms_db=-17.5,
        control_bias=EnhancementControls(
            pre_gain_db=-0.4,
            bass_gain_db=0.35,
            presence_gain_db=0.45,
            air_gain_db=0.25,
            stereo_width=1.03,
            compressor_threshold_db=-1.0,
            compressor_ratio=0.15,
            limiter_ceiling_db=-0.2,
        ),
    ),
    "apple_music": ServiceEnhancementProfile(
        name="apple_music",
        display_name="Apple Music",
        description=(
            "Preserves lossless headroom while applying gentle clarity and "
            "headphone-friendly width."
        ),
        target_rms_db=-18.5,
        control_bias=EnhancementControls(
            pre_gain_db=-0.7,
            bass_gain_db=0.15,
            presence_gain_db=0.30,
            air_gain_db=0.45,
            stereo_width=1.02,
            compressor_threshold_db=1.0,
            compressor_ratio=-0.15,
            limiter_ceiling_db=-0.5,
        ),
    ),
    "youtube_music": ServiceEnhancementProfile(
        name="youtube_music",
        display_name="YouTube Music",
        description=(
            "Reduces perceived loudness jumps and restores intelligibility for "
            "browser or app output with varied source quality."
        ),
        target_rms_db=-16.5,
        control_bias=EnhancementControls(
            pre_gain_db=-0.2,
            bass_gain_db=0.25,
            presence_gain_db=0.65,
            air_gain_db=0.20,
            stereo_width=1.01,
            compressor_threshold_db=-2.0,
            compressor_ratio=0.35,
            limiter_ceiling_db=-0.3,
        ),
    ),
}

ALIASES = {
    "apple": "apple_music",
    "apple music": "apple_music",
    "apple-music": "apple_music",
    "applemusic": "apple_music",
    "youtube": "youtube_music",
    "youtube music": "youtube_music",
    "youtube-music": "youtube_music",
    "youtubemusic": "youtube_music",
    "ytmusic": "youtube_music",
}


def get_service_profile(name: str | None = None) -> ServiceEnhancementProfile:
    if name is None:
        return NEUTRAL_PROFILE
    normalized = ALIASES.get(name.strip().lower(), name.strip().lower())
    normalized = normalized.replace(" ", "_").replace("-", "_")
    normalized = ALIASES.get(normalized, normalized)
    try:
        return SERVICE_PROFILES[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(SERVICE_PROFILES))
        raise ValueError(f"Unknown service profile '{name}'. Available profiles: {available}") from exc
