from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class EnhancementProfile:
    name: str
    target_backend: str
    normalize_peak: float
    compressor_threshold: float
    compressor_ratio: float
    makeup_gain: float
    soft_clip_drive: float
    low_shelf_gain_db: float
    presence_gain_db: float
    air_gain_db: float
    stereo_width: float
    vocal_center_focus: float
    transient_restore: float
    description: str


@dataclass(frozen=True)
class ServiceProfile:
    slug: str
    name: str
    loudness_trim: float
    presence_bias: float
    air_bias: float
    transient_bias: float
    notes: str


PROFILES: dict[str, EnhancementProfile] = {
    "balanced": EnhancementProfile(
        name="balanced",
        target_backend="cpu",
        normalize_peak=0.88,
        compressor_threshold=0.44,
        compressor_ratio=2.2,
        makeup_gain=1.04,
        soft_clip_drive=1.04,
        low_shelf_gain_db=0.8,
        presence_gain_db=0.9,
        air_gain_db=0.6,
        stereo_width=1.04,
        vocal_center_focus=1.03,
        transient_restore=0.08,
        description="natural loudness, clarity, and safe peak control",
    ),
    "snapdragon-x-npu": EnhancementProfile(
        name="snapdragon-x-npu",
        target_backend="onnxruntime-qnn",
        normalize_peak=0.91,
        compressor_threshold=0.38,
        compressor_ratio=2.8,
        makeup_gain=1.10,
        soft_clip_drive=1.10,
        low_shelf_gain_db=1.1,
        presence_gain_db=1.8,
        air_gain_db=1.4,
        stereo_width=1.12,
        vocal_center_focus=1.08,
        transient_restore=0.16,
        description="low-latency Snapdragon X NPU target with adaptive clarity",
    ),
    "holographic-vocal-stage": EnhancementProfile(
        name="holographic-vocal-stage",
        target_backend="onnxruntime-qnn",
        normalize_peak=0.90,
        compressor_threshold=0.34,
        compressor_ratio=3.0,
        makeup_gain=1.09,
        soft_clip_drive=1.08,
        low_shelf_gain_db=0.9,
        presence_gain_db=2.4,
        air_gain_db=2.0,
        stereo_width=1.30,
        vocal_center_focus=1.16,
        transient_restore=0.22,
        description="instrument separation, forward vocal image, and wider depth",
    ),
}


SERVICE_PROFILES: dict[str, ServiceProfile] = {
    "spotify": ServiceProfile(
        slug="spotify",
        name="Spotify",
        loudness_trim=0.95,
        presence_bias=0.15,
        air_bias=0.10,
        transient_bias=0.04,
        notes="compensates for normalized streams with conservative peak headroom",
    ),
    "apple-music": ServiceProfile(
        slug="apple-music",
        name="Apple Music",
        loudness_trim=1.02,
        presence_bias=0.05,
        air_bias=0.16,
        transient_bias=0.03,
        notes="keeps lossless output open while lifting ambience gently",
    ),
    "youtube-music": ServiceProfile(
        slug="youtube-music",
        name="YouTube Music",
        loudness_trim=0.92,
        presence_bias=0.22,
        air_bias=0.06,
        transient_bias=0.08,
        notes="reduces service-to-service loudness jumps and restores attack cues",
    ),
}


def available_profiles() -> tuple[str, ...]:
    return tuple(sorted(PROFILES))


def available_services() -> tuple[str, ...]:
    return tuple(sorted(SERVICE_PROFILES))


def get_profile(name: str) -> EnhancementProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        available = ", ".join(available_profiles())
        raise ValueError(f"unknown profile '{name}'. Available profiles: {available}") from exc


def get_service_profile(slug: str) -> ServiceProfile:
    try:
        return SERVICE_PROFILES[slug]
    except KeyError as exc:
        available = ", ".join(available_services())
        raise ValueError(f"unknown service '{slug}'. Available services: {available}") from exc


def adapt_profile_for_service(
    profile: EnhancementProfile,
    service: ServiceProfile,
) -> EnhancementProfile:
    return replace(
        profile,
        name=f"{profile.name}+{service.slug}",
        normalize_peak=max(0.70, min(0.96, profile.normalize_peak * service.loudness_trim)),
        presence_gain_db=profile.presence_gain_db + service.presence_bias,
        air_gain_db=profile.air_gain_db + service.air_bias,
        transient_restore=profile.transient_restore + service.transient_bias,
    )


def mix_profiles(
    profile: EnhancementProfile,
    services: tuple[ServiceProfile, ...],
) -> EnhancementProfile:
    effective = profile
    for service in services:
        effective = adapt_profile_for_service(effective, service)
    return effective
