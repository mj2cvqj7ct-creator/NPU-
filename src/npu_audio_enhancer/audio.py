from __future__ import annotations

import math
import struct
import wave
from dataclasses import dataclass
from pathlib import Path

from .profiles import EnhancementProfile, get_profile


SERVICE_PRESETS = {
    "spotify": {
        "display": "Spotify",
        "target_rms_offset": -0.006,
        "air_compensation": 0.012,
        "transient_compensation": 0.018,
        "stereo_compensation": 0.015,
    },
    "apple-music": {
        "display": "Apple Music",
        "target_rms_offset": 0.002,
        "air_compensation": 0.006,
        "transient_compensation": 0.010,
        "stereo_compensation": 0.004,
    },
    "youtube-music": {
        "display": "YouTube Music",
        "target_rms_offset": -0.010,
        "air_compensation": 0.018,
        "transient_compensation": 0.014,
        "stereo_compensation": 0.010,
    },
}


@dataclass(frozen=True)
class AudioBuffer:
    sample_rate: int
    channels: int
    samples: list[float]


@dataclass(frozen=True)
class EnhancementReport:
    profile: EnhancementProfile
    samples: int
    input_peak: float
    output_peak: float
    input_rms: float
    output_rms: float
    service: str


def read_wav(path: str | Path) -> AudioBuffer:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        raw = wav_file.readframes(frame_count)

    if sample_width != 2:
        raise ValueError("Only 16-bit PCM WAV files are supported")

    values = struct.unpack(f"<{len(raw) // 2}h", raw)
    return AudioBuffer(
        sample_rate=sample_rate,
        channels=channels,
        samples=[max(-1.0, min(1.0, value / 32768.0)) for value in values],
    )


def write_wav(path: str | Path, buffer: AudioBuffer) -> None:
    int_samples = [
        int(max(-1.0, min(1.0, sample)) * 32767)
        for sample in buffer.samples
    ]
    raw = struct.pack(f"<{len(int_samples)}h", *int_samples)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(buffer.channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(buffer.sample_rate)
        wav_file.writeframes(raw)


def generate_demo_buffer(duration_seconds: float = 1.0, sample_rate: int = 48_000) -> AudioBuffer:
    samples: list[float] = []
    total_frames = int(duration_seconds * sample_rate)
    for frame in range(total_frames):
        t = frame / sample_rate
        bass = 0.38 * math.sin(2 * math.pi * 110 * t)
        vocal = 0.22 * math.sin(2 * math.pi * 440 * t)
        air = 0.08 * math.sin(2 * math.pi * 4_800 * t)
        transient = 0.42 if frame % 9_600 < 80 else 0.0
        mixed = bass + vocal + air + transient
        samples.extend([mixed, mixed * 0.96])
    return AudioBuffer(sample_rate=sample_rate, channels=2, samples=samples)


def generate_demo_wav(path: str | Path) -> None:
    write_wav(path, generate_demo_buffer())


def peak(samples: list[float]) -> float:
    return max((abs(sample) for sample in samples), default=0.0)


def rms(samples: list[float]) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def enhance_audio(
    buffer: AudioBuffer,
    profile: EnhancementProfile,
    services: str | tuple[str, ...] = "auto",
) -> AudioBuffer:
    input_peak = peak(buffer.samples)
    if input_peak == 0:
        return buffer

    preset = _combined_service_preset(services)
    target_rms = max(0.08, profile.target_rms + float(preset["target_rms_offset"]))
    current_rms = rms(buffer.samples) or target_rms
    rms_gain = min(2.4, target_rms / current_rms)
    peak_gain = profile.normalize_peak / input_peak
    gain = min(rms_gain, peak_gain)
    normalized = [sample * gain for sample in buffer.samples]
    processed = _apply_spectral_focus(normalized, buffer.channels, profile)
    processed = _apply_service_compensation(processed, buffer.channels, profile, preset)
    processed = _apply_multiband_dynamics(processed, buffer.channels, profile)
    processed = [_compress_sample(sample, profile) for sample in processed]
    processed = _apply_stereo_image(processed, buffer.channels, profile)
    processed = _apply_neural_detail_surrogate(processed, buffer.channels, profile)
    clipped = [
        math.tanh(sample * profile.soft_clip_drive) / math.tanh(profile.soft_clip_drive)
        for sample in processed
    ]
    output_peak = peak(clipped)
    if output_peak > profile.normalize_peak:
        clipped = [
            sample * (profile.normalize_peak / output_peak)
            for sample in clipped
        ]

    return AudioBuffer(
        sample_rate=buffer.sample_rate,
        channels=buffer.channels,
        samples=clipped,
    )


def enhance_wav(
    input_path: str | Path,
    output_path: str | Path,
    profile_name: str,
    services: str | tuple[str, ...] = "auto",
) -> EnhancementReport:
    profile = get_profile(profile_name)
    source = read_wav(input_path)
    enhanced = enhance_audio(source, profile, services=services)
    write_wav(output_path, enhanced)
    return EnhancementReport(
        profile=profile,
        samples=len(source.samples),
        input_peak=peak(source.samples),
        output_peak=peak(enhanced.samples),
        input_rms=rms(source.samples),
        output_rms=rms(enhanced.samples),
        service=", ".join(_service_display_names(services)),
    )


def _compress_sample(sample: float, profile: EnhancementProfile) -> float:
    sign = -1.0 if sample < 0 else 1.0
    magnitude = abs(sample)
    if magnitude <= profile.compressor_threshold:
        return sample * profile.makeup_gain

    excess = magnitude - profile.compressor_threshold
    compressed = profile.compressor_threshold + (excess / profile.compressor_ratio)
    return sign * compressed * profile.makeup_gain


def _apply_spectral_focus(samples: list[float], channels: int, profile: EnhancementProfile) -> list[float]:
    if channels <= 0:
        return samples

    processed = [0.0] * len(samples)
    low_state = [0.0] * channels
    vocal_state = [0.0] * channels
    previous = [0.0] * channels
    low_alpha = 0.055
    vocal_alpha = 0.22
    separation = profile.instrument_separation

    for index, sample in enumerate(samples):
        channel = index % channels
        low_state[channel] += low_alpha * (sample - low_state[channel])
        vocal_state[channel] += vocal_alpha * (sample - vocal_state[channel])
        low = low_state[channel]
        vocal_band = vocal_state[channel] - low
        air_band = sample - vocal_state[channel]
        transient = sample - previous[channel]
        previous[channel] = sample

        processed[index] = (
            low * profile.bass_tightness
            + vocal_band * profile.vocal_presence
            + air_band * (1.0 + profile.air_lift)
            + transient * (profile.transient_snap - 1.0) * 0.35
            + (vocal_band - low) * separation * 0.12
        )

    return processed


def _apply_service_compensation(
    samples: list[float],
    channels: int,
    profile: EnhancementProfile,
    preset: dict[str, float | str],
) -> list[float]:
    if channels <= 0:
        return samples

    air = float(preset["air_compensation"])
    transient = float(preset["transient_compensation"])
    compensated: list[float] = []
    previous = [0.0] * channels
    for index, sample in enumerate(samples):
        channel = index % channels
        delta = sample - previous[channel]
        previous[channel] = sample
        restored = sample + delta * transient * (1.0 + profile.deartifact)
        compensated.append(restored * (1.0 + air * profile.neural_detail))
    return compensated


def _apply_multiband_dynamics(samples: list[float], channels: int, profile: EnhancementProfile) -> list[float]:
    if channels <= 0 or profile.loudness_guard <= 0:
        return samples

    processed = [0.0] * len(samples)
    low_state = [0.0] * channels
    mid_state = [0.0] * channels
    low_alpha = 0.040
    mid_alpha = 0.180
    guard = profile.loudness_guard
    for index, sample in enumerate(samples):
        channel = index % channels
        low_state[channel] += low_alpha * (sample - low_state[channel])
        mid_state[channel] += mid_alpha * (sample - mid_state[channel])
        low = low_state[channel]
        mid = mid_state[channel] - low
        high = sample - mid_state[channel]
        low = _soft_knee(low * profile.bass_tightness, 0.54, 1.8 + guard)
        mid = _soft_knee(mid * profile.vocal_presence, 0.46, 1.5 + guard)
        high = _soft_knee(high * (1.0 + profile.air_lift), 0.38, 1.4 + guard)
        processed[index] = low + mid + high
    return processed


def _apply_stereo_image(samples: list[float], channels: int, profile: EnhancementProfile) -> list[float]:
    if channels != 2:
        return samples

    enhanced: list[float] = []
    width = profile.stereo_width
    center = profile.center_focus
    air = profile.air_lift
    for index in range(0, len(samples), 2):
        left = samples[index]
        right = samples[index + 1]
        mid = (left + right) * 0.5 * center
        side = (left - right) * 0.5 * width
        shimmer = side * air
        enhanced.extend([mid + side + shimmer, mid - side - shimmer])
    return enhanced


def _apply_neural_detail_surrogate(samples: list[float], channels: int, profile: EnhancementProfile) -> list[float]:
    if channels <= 0 or profile.neural_detail <= 0:
        return samples

    detailed: list[float] = []
    previous = [0.0] * channels
    for index, sample in enumerate(samples):
        channel = index % channels
        detail = sample - previous[channel]
        previous[channel] = sample
        detailed.append(sample + detail * profile.neural_detail * 0.10)
    return detailed


def _soft_knee(sample: float, threshold: float, ratio: float) -> float:
    magnitude = abs(sample)
    if magnitude <= threshold:
        return sample
    sign = -1.0 if sample < 0 else 1.0
    excess = magnitude - threshold
    return sign * (threshold + excess / ratio)


def _service_preset(service: str) -> dict[str, float | str]:
    key = _normalize_service_key(service)
    if key == "auto":
        return {
            "display": "Auto service profile",
            "target_rms_offset": -0.004,
            "air_compensation": 0.012,
            "transient_compensation": 0.014,
            "stereo_compensation": 0.010,
        }
    try:
        return SERVICE_PRESETS[key]
    except KeyError as exc:
        available = ", ".join(("auto", *sorted(SERVICE_PRESETS)))
        raise ValueError(f"unknown service '{service}'. Available services: {available}") from exc


def _combined_service_preset(services: str | tuple[str, ...]) -> dict[str, float | str]:
    names = _service_names(services)
    presets = [_service_preset(name) for name in names]
    if not presets:
        presets = [_service_preset("auto")]
    return {
        "display": ", ".join(str(preset["display"]) for preset in presets),
        "target_rms_offset": _average(presets, "target_rms_offset"),
        "air_compensation": _average(presets, "air_compensation"),
        "transient_compensation": _average(presets, "transient_compensation"),
        "stereo_compensation": _average(presets, "stereo_compensation"),
    }


def _service_display_names(services: str | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(_service_preset(name)["display"]) for name in _service_names(services))


def _service_names(services: str | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(services, str):
        return (services,)
    return tuple(services)


def _average(presets: list[dict[str, float | str]], key: str) -> float:
    return sum(float(preset[key]) for preset in presets) / len(presets)


def _normalize_service_key(service: str) -> str:
    normalized = service.strip().lower().replace("_", "-").replace(" ", "-")
    aliases = {
        "apple": "apple-music",
        "applemusic": "apple-music",
        "itunes": "apple-music",
        "youtube": "youtube-music",
        "youtubemusic": "youtube-music",
        "ytmusic": "youtube-music",
    }
    return aliases.get(normalized, normalized)
