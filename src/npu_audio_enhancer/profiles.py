from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnhancementProfile:
    name: str
    service: str
    target_backend: str
    frame_ms: float
    true_peak_target: float
    max_frame_gain: float
    compressor_threshold: float
    compressor_ratio: float
    makeup_gain: float
    soft_clip_drive: float
    clarity_weight: float
    bass_weight: float
    air_weight: float
    transient_weight: float
    stereo_width: float
    vocal_focus: float
    description: str


PROFILES: dict[str, EnhancementProfile] = {
    "snapdragon-x-npu": EnhancementProfile(
        name="snapdragon-x-npu",
        service="system",
        target_backend="onnxruntime-qnn",
        frame_ms=10.0,
        true_peak_target=0.92,
        max_frame_gain=2.8,
        compressor_threshold=0.40,
        compressor_ratio=2.6,
        makeup_gain=1.06,
        soft_clip_drive=1.08,
        clarity_weight=0.26,
        bass_weight=0.18,
        air_weight=0.16,
        transient_weight=0.18,
        stereo_width=1.08,
        vocal_focus=1.06,
        description="balanced low-latency enhancement for Snapdragon X NPU",
    ),
    "spotify-npu": EnhancementProfile(
        name="spotify-npu",
        service="Spotify",
        target_backend="onnxruntime-qnn",
        frame_ms=10.0,
        true_peak_target=0.91,
        max_frame_gain=2.6,
        compressor_threshold=0.37,
        compressor_ratio=2.9,
        makeup_gain=1.08,
        soft_clip_drive=1.12,
        clarity_weight=0.34,
        bass_weight=0.15,
        air_weight=0.18,
        transient_weight=0.28,
        stereo_width=1.11,
        vocal_focus=1.10,
        description="codec-recovery clarity, transient lift, and volume smoothing",
    ),
    "apple-lossless-npu": EnhancementProfile(
        name="apple-lossless-npu",
        service="Apple Music",
        target_backend="onnxruntime-qnn",
        frame_ms=10.0,
        true_peak_target=0.90,
        max_frame_gain=2.2,
        compressor_threshold=0.44,
        compressor_ratio=2.1,
        makeup_gain=1.03,
        soft_clip_drive=1.04,
        clarity_weight=0.18,
        bass_weight=0.12,
        air_weight=0.20,
        transient_weight=0.12,
        stereo_width=1.06,
        vocal_focus=1.05,
        description="lossless-first spatial polish with conservative dynamics",
    ),
    "youtube-music-npu": EnhancementProfile(
        name="youtube-music-npu",
        service="YouTube Music",
        target_backend="onnxruntime-qnn",
        frame_ms=10.0,
        true_peak_target=0.89,
        max_frame_gain=2.4,
        compressor_threshold=0.35,
        compressor_ratio=3.2,
        makeup_gain=1.07,
        soft_clip_drive=1.10,
        clarity_weight=0.30,
        bass_weight=0.12,
        air_weight=0.14,
        transient_weight=0.25,
        stereo_width=1.07,
        vocal_focus=1.13,
        description="browser-video loudness leveling and vocal intelligibility",
    ),
}


SERVICE_PROFILE_ALIASES = {
    "spotify": "spotify-npu",
    "apple": "apple-lossless-npu",
    "applemusic": "apple-lossless-npu",
    "apple_music": "apple-lossless-npu",
    "apple-music": "apple-lossless-npu",
    "youtube": "youtube-music-npu",
    "youtubemusic": "youtube-music-npu",
    "youtube_music": "youtube-music-npu",
    "youtube-music": "youtube-music-npu",
    "ytmusic": "youtube-music-npu",
}


def get_profile(name: str) -> EnhancementProfile:
    resolved = SERVICE_PROFILE_ALIASES.get(_compact(name), name)
    try:
        return PROFILES[resolved]
    except KeyError as exc:
        available = ", ".join(sorted(PROFILES))
        raise ValueError(f"unknown profile '{name}'. Available profiles: {available}") from exc


def service_profile_name(service: str) -> str:
    try:
        return SERVICE_PROFILE_ALIASES[_compact(service)]
    except KeyError as exc:
        available = ", ".join(sorted(SERVICE_PROFILE_ALIASES))
        raise ValueError(f"unknown service '{service}'. Available aliases: {available}") from exc


def available_profiles() -> tuple[str, ...]:
    return tuple(sorted(PROFILES))


def _compact(value: str) -> str:
    return value.strip().lower().replace(" ", "").replace("-", "_")
