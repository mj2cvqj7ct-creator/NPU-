"""Service-aware tuning profiles for music-service PCM enhancement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dsp import AudioFeatures, EnhancementControls


@dataclass(frozen=True)
class ServiceProfile:
    """DSP and NPU blending policy for a music service output stream."""

    name: str
    display_name: str
    target_rms_db: float
    npu_mix: float
    control_bias: EnhancementControls
    description: str

    def apply(self, controls: EnhancementControls, features: AudioFeatures) -> EnhancementControls:
        clipping_guard = features.clipping_ratio > 0.001 or features.peak_db > -0.2
        transient_bias = self.control_bias.transient_gain_db
        if features.transient_density > 0.18:
            transient_bias *= 0.5
        return EnhancementControls(
            pre_gain_db=float(np.clip(controls.pre_gain_db, -6.0, 6.0)),
            bass_gain_db=float(np.clip(controls.bass_gain_db + self.control_bias.bass_gain_db, -3.0, 3.0)),
            presence_gain_db=float(
                np.clip(controls.presence_gain_db + self.control_bias.presence_gain_db, -3.0, 3.0)
            ),
            air_gain_db=float(np.clip(controls.air_gain_db + self.control_bias.air_gain_db, -3.0, 3.0)),
            stereo_width=float(np.clip(controls.stereo_width * self.control_bias.stereo_width, 0.5, 1.4)),
            transient_gain_db=float(np.clip(controls.transient_gain_db + transient_bias, -1.5, 2.0)),
            compressor_threshold_db=float(
                min(controls.compressor_threshold_db, self.control_bias.compressor_threshold_db)
            ),
            compressor_ratio=float(
                max(controls.compressor_ratio, self.control_bias.compressor_ratio, 2.2 if clipping_guard else 1.0)
            ),
            limiter_ceiling_db=float(min(controls.limiter_ceiling_db, self.control_bias.limiter_ceiling_db)),
        )


PROFILES: dict[str, ServiceProfile] = {
    "auto": ServiceProfile(
        name="auto",
        display_name="Auto music service",
        target_rms_db=-17.5,
        npu_mix=0.45,
        control_bias=EnhancementControls(
            bass_gain_db=0.2,
            presence_gain_db=0.25,
            air_gain_db=0.2,
            stereo_width=1.03,
            transient_gain_db=0.2,
            compressor_threshold_db=-18.0,
            compressor_ratio=1.65,
            limiter_ceiling_db=-1.0,
        ),
        description="Balanced cross-service tuning for Spotify, Apple Music, and YouTube Music.",
    ),
    "balanced": ServiceProfile(
        name="balanced",
        display_name="Balanced system audio",
        target_rms_db=-18.0,
        npu_mix=0.35,
        control_bias=EnhancementControls(),
        description="Natural correction for mixed app output.",
    ),
    "spotify": ServiceProfile(
        name="spotify",
        display_name="Spotify",
        target_rms_db=-17.0,
        npu_mix=0.48,
        control_bias=EnhancementControls(
            bass_gain_db=0.35,
            presence_gain_db=0.25,
            air_gain_db=0.2,
            stereo_width=1.04,
            transient_gain_db=0.35,
            compressor_threshold_db=-17.0,
            compressor_ratio=1.65,
            limiter_ceiling_db=-1.0,
        ),
        description="Restores transient punch and clarity after loudness-normalized streams.",
    ),
    "apple-music": ServiceProfile(
        name="apple-music",
        display_name="Apple Music",
        target_rms_db=-18.5,
        npu_mix=0.42,
        control_bias=EnhancementControls(
            bass_gain_db=0.15,
            presence_gain_db=0.35,
            air_gain_db=0.3,
            stereo_width=1.03,
            transient_gain_db=0.15,
            compressor_threshold_db=-18.0,
            compressor_ratio=1.45,
            limiter_ceiling_db=-1.0,
        ),
        description="Keeps lossless/dolby-capable output transparent while improving focus.",
    ),
    "youtube-music": ServiceProfile(
        name="youtube-music",
        display_name="YouTube Music",
        target_rms_db=-16.5,
        npu_mix=0.52,
        control_bias=EnhancementControls(
            bass_gain_db=0.25,
            presence_gain_db=0.45,
            air_gain_db=0.35,
            stereo_width=1.02,
            transient_gain_db=0.25,
            compressor_threshold_db=-19.0,
            compressor_ratio=1.85,
            limiter_ceiling_db=-1.2,
        ),
        description="Smooths level variance and repairs dull or over-compressed browser output.",
    ),
    "snapdragon-x-npu": ServiceProfile(
        name="snapdragon-x-npu",
        display_name="Snapdragon X NPU studio",
        target_rms_db=-17.5,
        npu_mix=0.65,
        control_bias=EnhancementControls(
            bass_gain_db=0.35,
            presence_gain_db=0.45,
            air_gain_db=0.45,
            stereo_width=1.08,
            transient_gain_db=0.45,
            compressor_threshold_db=-18.0,
            compressor_ratio=1.6,
            limiter_ceiling_db=-1.0,
        ),
        description="Aggressive NPU-assisted enhancement target for Snapdragon X ARM64 PCs.",
    ),
}


def get_profile(name: str | None) -> ServiceProfile:
    key = (name or "balanced").strip().lower()
    try:
        return PROFILES[key]
    except KeyError as exc:
        available = ", ".join(available_profiles())
        raise ValueError(f"unknown service profile '{name}'. Available profiles: {available}") from exc


get_service_profile = get_profile


def available_profiles() -> tuple[str, ...]:
    return tuple(sorted(PROFILES))


PROFILE_NAMES = available_profiles()
