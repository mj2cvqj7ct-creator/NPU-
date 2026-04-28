from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnhancementProfile:
    """Service-aware tuning without touching app internals or protected streams."""

    name: str
    target_backend: str
    target_lufs: float
    true_peak_ceiling: float
    max_gain: float
    compressor_threshold: float
    compressor_ratio: float
    makeup_gain: float
    bass_lift_db: float
    presence_lift_db: float
    air_lift_db: float
    stereo_width: float
    center_focus: float
    max_stereo_width: float
    transient_restore: float
    description: str


PROFILES: dict[str, EnhancementProfile] = {
    "balanced": EnhancementProfile(
        name="balanced",
        target_backend="cpu",
        target_lufs=-16.0,
        true_peak_ceiling=0.89,
        max_gain=4.0,
        compressor_threshold=0.48,
        compressor_ratio=2.2,
        makeup_gain=1.04,
        bass_lift_db=0.8,
        presence_lift_db=0.9,
        air_lift_db=0.7,
        stereo_width=1.04,
        center_focus=1.02,
        max_stereo_width=1.16,
        transient_restore=0.05,
        description="general low-latency clarity for mixed streaming sources",
    ),
    "spotify": EnhancementProfile(
        name="spotify",
        target_backend="onnxruntime-qnn",
        target_lufs=-15.5,
        true_peak_ceiling=0.90,
        max_gain=4.5,
        compressor_threshold=0.44,
        compressor_ratio=2.6,
        makeup_gain=1.06,
        bass_lift_db=0.6,
        presence_lift_db=1.2,
        air_lift_db=0.8,
        stereo_width=1.05,
        center_focus=1.04,
        max_stereo_width=1.18,
        transient_restore=0.07,
        description="restores edge definition after high-efficiency streaming compression",
    ),
    "apple-music": EnhancementProfile(
        name="apple-music",
        target_backend="onnxruntime-qnn",
        target_lufs=-17.0,
        true_peak_ceiling=0.92,
        max_gain=3.0,
        compressor_threshold=0.52,
        compressor_ratio=1.8,
        makeup_gain=1.02,
        bass_lift_db=0.4,
        presence_lift_db=0.7,
        air_lift_db=0.5,
        stereo_width=1.03,
        center_focus=1.01,
        max_stereo_width=1.12,
        transient_restore=0.04,
        description="gentle post-processing for lossless or high-bitrate playback",
    ),
    "youtube-music": EnhancementProfile(
        name="youtube-music",
        target_backend="onnxruntime-qnn",
        target_lufs=-14.5,
        true_peak_ceiling=0.88,
        max_gain=5.0,
        compressor_threshold=0.40,
        compressor_ratio=2.9,
        makeup_gain=1.08,
        bass_lift_db=0.9,
        presence_lift_db=1.4,
        air_lift_db=1.1,
        stereo_width=1.02,
        center_focus=1.05,
        max_stereo_width=1.14,
        transient_restore=0.08,
        description="stronger volume leveling for varied upload mastering",
    ),
    "snapdragon-x-npu": EnhancementProfile(
        name="snapdragon-x-npu",
        target_backend="onnxruntime-qnn",
        target_lufs=-15.0,
        true_peak_ceiling=0.91,
        max_gain=4.5,
        compressor_threshold=0.42,
        compressor_ratio=2.8,
        makeup_gain=1.08,
        bass_lift_db=0.9,
        presence_lift_db=1.5,
        air_lift_db=1.0,
        stereo_width=1.10,
        center_focus=1.06,
        max_stereo_width=1.24,
        transient_restore=0.10,
        description="aggressive local enhancement intended for QNN-backed inference",
    ),
}


def get_profile(name: str) -> EnhancementProfile:
    key = name.strip().lower()
    try:
        return PROFILES[key]
    except KeyError as exc:
        available = ", ".join(available_profiles())
        raise ValueError(f"unknown profile '{name}'. Available profiles: {available}") from exc


def available_profiles() -> tuple[str, ...]:
    return tuple(sorted(PROFILES))
