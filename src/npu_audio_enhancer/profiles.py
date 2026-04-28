from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnhancementProfile:
    name: str
    target_backend: str
    normalize_peak: float
    target_rms: float
    compressor_threshold: float
    compressor_ratio: float
    makeup_gain: float
    soft_clip_drive: float
    stereo_width: float = 1.0
    center_focus: float = 1.0
    air_lift: float = 0.0
    bass_tightness: float = 1.0
    vocal_presence: float = 1.0
    instrument_separation: float = 0.0
    transient_snap: float = 1.0
    neural_detail: float = 0.0
    deartifact: float = 0.0
    loudness_guard: float = 0.0
    description: str = "balanced clarity"


@dataclass(frozen=True)
class ServiceTuning:
    name: str
    loudness_bias: float
    bass_bias: float
    vocal_bias: float
    air_bias: float
    transient_bias: float
    stereo_bias: float
    deartifact_bias: float
    notes: str


PROFILES = {
    "balanced": EnhancementProfile(
        name="balanced",
        target_backend="cpu",
        normalize_peak=0.88,
        target_rms=0.18,
        compressor_threshold=0.42,
        compressor_ratio=2.4,
        makeup_gain=1.08,
        soft_clip_drive=1.08,
        bass_tightness=1.02,
        vocal_presence=1.02,
        transient_snap=1.02,
        description="natural loudness and peak control",
    ),
    "snapdragon-x-npu": EnhancementProfile(
        name="snapdragon-x-npu",
        target_backend="onnxruntime-qnn",
        normalize_peak=0.92,
        target_rms=0.20,
        compressor_threshold=0.38,
        compressor_ratio=2.8,
        makeup_gain=1.12,
        soft_clip_drive=1.12,
        stereo_width=1.10,
        center_focus=1.05,
        air_lift=0.03,
        bass_tightness=1.08,
        vocal_presence=1.10,
        instrument_separation=0.08,
        transient_snap=1.06,
        neural_detail=0.14,
        deartifact=0.10,
        loudness_guard=0.08,
        description="low-latency Snapdragon X NPU target",
    ),
    "holographic-vocal-stage": EnhancementProfile(
        name="holographic-vocal-stage",
        target_backend="onnxruntime-qnn",
        normalize_peak=0.90,
        target_rms=0.19,
        compressor_threshold=0.34,
        compressor_ratio=3.0,
        makeup_gain=1.10,
        soft_clip_drive=1.10,
        stereo_width=1.32,
        center_focus=1.16,
        air_lift=0.08,
        bass_tightness=1.14,
        vocal_presence=1.22,
        instrument_separation=0.18,
        transient_snap=1.10,
        neural_detail=0.22,
        deartifact=0.12,
        loudness_guard=0.12,
        description="holographic imaging, instrument separation, and vocal focus",
    ),
    "streaming-master": EnhancementProfile(
        name="streaming-master",
        target_backend="onnxruntime-qnn",
        normalize_peak=0.91,
        target_rms=0.205,
        compressor_threshold=0.36,
        compressor_ratio=2.6,
        makeup_gain=1.11,
        soft_clip_drive=1.09,
        stereo_width=1.18,
        center_focus=1.10,
        air_lift=0.055,
        bass_tightness=1.12,
        vocal_presence=1.16,
        instrument_separation=0.14,
        transient_snap=1.12,
        neural_detail=0.20,
        deartifact=0.16,
        loudness_guard=0.16,
        description="service-aware NPU mastering for Spotify, Apple Music, and YouTube Music",
    ),
}


SERVICE_TUNINGS = {
    "generic": ServiceTuning(
        name="generic",
        loudness_bias=1.0,
        bass_bias=1.0,
        vocal_bias=1.0,
        air_bias=1.0,
        transient_bias=1.0,
        stereo_bias=1.0,
        deartifact_bias=1.0,
        notes="neutral post-processing for unknown PCM sources",
    ),
    "spotify": ServiceTuning(
        name="spotify",
        loudness_bias=0.97,
        bass_bias=1.04,
        vocal_bias=1.03,
        air_bias=1.02,
        transient_bias=1.05,
        stereo_bias=1.02,
        deartifact_bias=1.12,
        notes="counteracts loudness-normalized lossy streams with transient and deartifact focus",
    ),
    "apple-music": ServiceTuning(
        name="apple-music",
        loudness_bias=1.0,
        bass_bias=1.02,
        vocal_bias=1.02,
        air_bias=1.04,
        transient_bias=1.02,
        stereo_bias=1.01,
        deartifact_bias=0.92,
        notes="keeps lossless streams clean while adding headphone-aware air and focus",
    ),
    "youtube-music": ServiceTuning(
        name="youtube-music",
        loudness_bias=0.95,
        bass_bias=1.01,
        vocal_bias=1.06,
        air_bias=1.0,
        transient_bias=1.04,
        stereo_bias=0.98,
        deartifact_bias=1.18,
        notes="smooths codec variance and inconsistent video loudness",
    ),
}


def get_profile(name: str) -> EnhancementProfile:
    try:
        return PROFILES[name]
    except KeyError as exc:
        available = ", ".join(sorted(PROFILES))
        raise ValueError(f"unknown profile '{name}'. Available profiles: {available}") from exc


def available_profiles() -> tuple[str, ...]:
    return tuple(sorted(PROFILES))


def get_service_tuning(name: str | None) -> ServiceTuning:
    key = _normalize_service_name(name)
    try:
        return SERVICE_TUNINGS[key]
    except KeyError as exc:
        available = ", ".join(sorted(SERVICE_TUNINGS))
        raise ValueError(f"unknown service '{name}'. Available services: {available}") from exc


def available_services() -> tuple[str, ...]:
    return tuple(sorted(SERVICE_TUNINGS))


def _normalize_service_name(name: str | None) -> str:
    if not name:
        return "generic"
    normalized = name.strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "apple": "apple-music",
        "applemusic": "apple-music",
        "youtube": "youtube-music",
        "ytmusic": "youtube-music",
        "youtube-music": "youtube-music",
    }
    return aliases.get(normalized, normalized)
