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


PROFILES = {
    "balanced": EnhancementProfile(
        name="balanced",
        target_backend="cpu",
        normalize_peak=0.88,
        compressor_threshold=0.42,
        compressor_ratio=2.4,
        makeup_gain=1.08,
        soft_clip_drive=1.08,
    ),
    "snapdragon-x-npu": EnhancementProfile(
        name="snapdragon-x-npu",
        target_backend="onnxruntime-qnn",
        normalize_peak=0.92,
        compressor_threshold=0.38,
        compressor_ratio=2.8,
        makeup_gain=1.12,
        soft_clip_drive=1.12,
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
