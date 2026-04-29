from __future__ import annotations

import math
import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .npu import EnhancementFeatures, HeuristicNpuModel
from .profiles import EnhancementProfile, get_profile


@dataclass(frozen=True)
class AudioBuffer:
    sample_rate: int
    channels: int
    samples: list[float]

    @property
    def frames(self) -> int:
        if self.channels <= 0:
            return 0
        return len(self.samples) // self.channels


@dataclass(frozen=True)
class EnhancementReport:
    profile: EnhancementProfile
    samples: int
    frames: int
    input_peak: float
    output_peak: float
    average_features: EnhancementFeatures

    @property
    def mean_clarity(self) -> float:
        return self.average_features.clarity

    @property
    def mean_warmth(self) -> float:
        return self.average_features.bass_tightness

    @property
    def mean_transient_lift(self) -> float:
        return self.average_features.transient_restore


class NpuModel(Protocol):
    def infer(self, samples: list[float], profile: EnhancementProfile) -> EnhancementFeatures:
        ...


def read_wav(path: str | Path) -> AudioBuffer:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        raw = wav_file.readframes(frame_count)

    if channels < 1:
        raise ValueError("WAV file must contain at least one channel")
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
        bass = 0.34 * math.sin(2 * math.pi * 82.41 * t)
        kick = 0.42 if frame % 12_000 < 55 else 0.0
        vocal = 0.23 * math.sin(2 * math.pi * 392.0 * t)
        guitar = 0.11 * math.sin(2 * math.pi * 1_760.0 * t)
        cymbal = 0.06 * math.sin(2 * math.pi * 8_800.0 * t)
        side_motion = 0.035 * math.sin(2 * math.pi * 0.7 * t)
        left = bass + kick + vocal + guitar + cymbal + side_motion
        right = bass + kick * 0.96 + vocal * 0.98 + guitar * 0.92 + cymbal - side_motion
        samples.extend([left, right])
    return AudioBuffer(sample_rate=sample_rate, channels=2, samples=samples)


def generate_demo_wav(path: str | Path) -> None:
    write_wav(path, generate_demo_buffer())


def peak(samples: list[float]) -> float:
    return max((abs(sample) for sample in samples), default=0.0)


def enhance_audio(
    buffer: AudioBuffer,
    profile: EnhancementProfile,
    npu_model: NpuModel | None = None,
    frame_ms: float | None = None,
) -> AudioBuffer:
    if buffer.channels <= 0:
        raise ValueError("Audio buffer must contain at least one channel")
    if len(buffer.samples) % buffer.channels != 0:
        raise ValueError("Audio samples must be aligned to channel count")
    if not buffer.samples:
        return buffer

    model = npu_model or HeuristicNpuModel()
    frame_size = max(1, int(buffer.sample_rate * (frame_ms or profile.frame_ms) / 1000))
    frame_samples = frame_size * buffer.channels
    output: list[float] = []

    for start in range(0, len(buffer.samples), frame_samples):
        chunk = buffer.samples[start:start + frame_samples]
        features = model.infer(chunk, profile)
        processed = _process_frame(chunk, buffer.channels, profile, features)
        output.extend(processed)

    return AudioBuffer(
        sample_rate=buffer.sample_rate,
        channels=buffer.channels,
        samples=output,
    )


def enhance_wav(
    input_path: str | Path,
    output_path: str | Path,
    profile_name: str,
    frame_ms: float | None = None,
) -> EnhancementReport:
    profile = get_profile(profile_name)
    source = read_wav(input_path)
    features = estimate_average_features(source, profile, frame_ms=frame_ms)
    enhanced = enhance_audio(source, profile, frame_ms=frame_ms)
    write_wav(output_path, enhanced)
    return EnhancementReport(
        profile=profile,
        samples=len(source.samples),
        frames=source.frames,
        input_peak=peak(source.samples),
        output_peak=peak(enhanced.samples),
        average_features=features,
    )


def estimate_average_features(
    buffer: AudioBuffer,
    profile: EnhancementProfile,
    npu_model: NpuModel | None = None,
    frame_ms: float | None = None,
) -> EnhancementFeatures:
    if not buffer.samples:
        return EnhancementFeatures.neutral()

    model = npu_model or HeuristicNpuModel()
    frame_size = max(1, int(buffer.sample_rate * (frame_ms or profile.frame_ms) / 1000))
    frame_samples = frame_size * buffer.channels
    feature_sum = EnhancementFeatures.zero()
    feature_count = 0
    for start in range(0, len(buffer.samples), frame_samples):
        chunk = buffer.samples[start:start + frame_samples]
        feature_sum = feature_sum + model.infer(chunk, profile)
        feature_count += 1
    return feature_sum.scale(1.0 / feature_count)


def _process_frame(
    samples: list[float],
    channels: int,
    profile: EnhancementProfile,
    features: EnhancementFeatures,
) -> list[float]:
    input_peak = peak(samples)
    if input_peak == 0.0:
        return list(samples)

    target = profile.true_peak_target
    normalized = [sample * min(target / input_peak, profile.max_frame_gain) for sample in samples]
    tonal = _apply_tonal_balance(normalized, channels, profile, features)
    compressed = [_compress_sample(sample, profile, features) for sample in tonal]
    imaged = _apply_stereo_image(compressed, channels, profile, features)
    protected = _apply_transient_protection(imaged, profile, features)
    return _limit_true_peak(protected, target)


def _apply_tonal_balance(
    samples: list[float],
    channels: int,
    profile: EnhancementProfile,
    features: EnhancementFeatures,
) -> list[float]:
    # Lightweight shelving approximation for CPU tests; the NPU path supplies
    # feature weights that can later drive a higher quality ONNX model.
    bass_gain = 1.0 + profile.bass_weight * (features.bass_tightness - 0.5)
    presence_gain = 1.0 + profile.clarity_weight * features.clarity * profile.vocal_focus
    air_gain = 1.0 + profile.air_weight * (1.0 - features.noise_floor)
    bass_gain = _clamp(bass_gain, 0.82, 1.18)
    presence_gain = _clamp(presence_gain, 0.94, 1.16)
    air_gain = _clamp(air_gain, 0.96, 1.14)

    filtered: list[float] = []
    bass_state = [0.0] * channels
    air_state = [0.0] * channels
    alpha_bass = 0.045
    alpha_air = 0.62
    for index, sample in enumerate(samples):
        channel = index % channels
        bass_state[channel] += alpha_bass * (sample - bass_state[channel])
        high = sample - air_state[channel]
        air_state[channel] += alpha_air * (sample - air_state[channel])
        mid = sample - bass_state[channel] - high
        filtered.append(
            bass_state[channel] * bass_gain
            + mid * presence_gain
            + high * air_gain
        )
    return filtered


def _compress_sample(
    sample: float,
    profile: EnhancementProfile,
    features: EnhancementFeatures,
) -> float:
    threshold = profile.compressor_threshold - (features.transient_restore * 0.05)
    ratio = profile.compressor_ratio + (features.transient_restore * 0.7)
    sign = -1.0 if sample < 0 else 1.0
    magnitude = abs(sample)
    if magnitude <= threshold:
        return sample * profile.makeup_gain

    excess = magnitude - threshold
    compressed = threshold + (excess / ratio)
    return sign * compressed * profile.makeup_gain


def _apply_stereo_image(
    samples: list[float],
    channels: int,
    profile: EnhancementProfile,
    features: EnhancementFeatures,
) -> list[float]:
    if channels != 2:
        return samples

    width = _clamp(profile.stereo_width + features.stereo_focus * 0.12, 0.86, 1.34)
    center = _clamp(profile.vocal_focus + features.vocal_presence * 0.10, 0.96, 1.22)
    enhanced: list[float] = []
    for index in range(0, len(samples), 2):
        left = samples[index]
        right = samples[index + 1]
        mid = (left + right) * 0.5 * center
        side = (left - right) * 0.5 * width
        enhanced.extend([mid + side, mid - side])
    return enhanced


def _apply_transient_protection(
    samples: list[float],
    profile: EnhancementProfile,
    features: EnhancementFeatures,
) -> list[float]:
    drive = profile.soft_clip_drive + features.transient_restore * profile.transient_weight * 0.35
    normalizer = math.tanh(drive)
    return [math.tanh(sample * drive) / normalizer for sample in samples]


def _limit_true_peak(samples: list[float], target: float) -> list[float]:
    output_peak = peak(samples)
    if output_peak <= target or output_peak == 0.0:
        return samples
    gain = target / output_peak
    return [sample * gain for sample in samples]


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
