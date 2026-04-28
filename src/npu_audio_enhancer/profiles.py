from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnhancementProfile:
    name: str
    target_backend: str
    normalize_peak: float
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
    description: str = "balanced clarity"


PROFILES = {
    "balanced": EnhancementProfile(
        name="balanced",
        target_backend="cpu",
        normalize_peak=0.88,
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
        description="low-latency Snapdragon X NPU target",
    ),
    "holographic-vocal-stage": EnhancementProfile(
        name="holographic-vocal-stage",
        target_backend="onnxruntime-qnn",
        normalize_peak=0.90,
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
        description="holographic imaging, instrument separation, and vocal focus",
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
